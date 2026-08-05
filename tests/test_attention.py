from __future__ import annotations

import os
import types
import unittest
from typing import ClassVar
from unittest import mock

import numpy as np
import torch
from PIL import Image

from vlm_attention_viz.attention import (
    AttentionCapture,
    build_session_metadata,
    discover_full_attention_layers,
    extract_attention,
    load_model,
    resolve_visual_grids,
    select_attention_head,
    validate_visual_token_groups,
)


class FakeTokenizer:
    all_special_ids: ClassVar[list[int]] = [10, 11, 12, 99]

    def convert_ids_to_tokens(self, ids):
        return [f"piece-{token_id}" for token_id in ids]

    def decode(self, ids, skip_special_tokens=False):
        self.skip_special_tokens = skip_special_tokens
        return f"decoded-{ids[0]}"


class MetadataTest(unittest.TestCase):
    def test_preserves_all_non_image_tokens_and_absolute_positions(self):
        full_ids = [10, 20, 99, 99, 11, 21, 21, 12]

        metadata = build_session_metadata(
            tokenizer=FakeTokenizer(),
            full_ids=full_ids,
            prompt_length=5,
            image_token_id=99,
        )

        self.assertEqual(
            [token.absolute_position for token in metadata.tokens if not token.is_image],
            [0, 1, 4, 5, 6, 7],
        )
        self.assertEqual(metadata.generated_positions.tolist(), [5, 6, 7])
        self.assertEqual(metadata.selectable_positions.tolist(), [0, 1, 4, 5, 6, 7])
        self.assertEqual(metadata.query_positions.tolist(), [0, 1, 4, 5, 6, 7])
        self.assertEqual(metadata.visual_key_positions.tolist(), [2, 3])
        self.assertEqual(metadata.context_key_positions.tolist(), [0, 1, 4, 5, 6, 7])
        np.testing.assert_array_equal(
            metadata.causal_context_mask,
            np.array(
                [
                    [False, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, False],
                ]
            ),
        )
        self.assertEqual(metadata.tokens[5].segment, "generated")
        self.assertEqual(metadata.tokens[6].raw_piece, "piece-21")
        self.assertTrue(metadata.tokens[7].is_special)

    def test_streaming_decode_produces_readable_contextual_token_labels(self):
        class StreamingTokenizer(FakeTokenizer):
            backend_tokenizer = object()

            def convert_ids_to_tokens(self, ids):
                pieces = {
                    10: "<|im_start|>",
                    20: "system",
                    21: "\u010a",
                    22: "\u0120You",
                    30: "byte-a",
                    31: "byte-b",
                    99: "<|image_pad|>",
                    12: "<|im_end|>",
                }
                return [pieces[token_id] for token_id in ids]

        chunks = {
            10: "<|im_start|>",
            20: "system",
            21: "\n",
            22: " You",
            30: None,
            31: "\u4e2d",
            99: "<|image_pad|>",
            12: "<|im_end|>",
        }

        class FakeDecodeStream:
            def __init__(self, skip_special_tokens=False):
                self.skip_special_tokens = skip_special_tokens

            def step(self, tokenizer, token_id):
                return chunks[token_id]

        decoder_module = types.SimpleNamespace(DecodeStream=FakeDecodeStream)
        with mock.patch.dict("sys.modules", {"tokenizers.decoders": decoder_module}):
            metadata = build_session_metadata(
                StreamingTokenizer(),
                [10, 20, 21, 22, 99, 30, 31, 12],
                prompt_length=5,
                image_token_id=99,
            )

        self.assertEqual(
            [token.decoded_preview for token in metadata.tokens],
            [
                "<|im_start|>",
                "system",
                "\n",
                " You",
                "<|image_pad|>",
                "",
                "\u4e2d",
                "<|im_end|>",
            ],
        )

    def test_rejects_empty_generation_and_missing_image_tokens(self):
        with self.assertRaisesRegex(ValueError, "generated token"):
            build_session_metadata(FakeTokenizer(), [1, 99], 2, 99)
        with self.assertRaisesRegex(ValueError, "image token"):
            build_session_metadata(FakeTokenizer(), [1, 2], 1, 99)

    def test_text_only_metadata_makes_every_token_selectable(self):
        metadata = build_session_metadata(FakeTokenizer(), [10, 11, 20], 2, None)

        self.assertEqual(metadata.visual_key_positions.tolist(), [])
        self.assertEqual(metadata.selectable_positions.tolist(), [0, 1, 2])
        self.assertEqual(metadata.query_positions.tolist(), [0, 1, 2])
        self.assertFalse(any(token.is_image for token in metadata.tokens))


class LayerDiscoveryTest(unittest.TestCase):
    def test_hybrid_model_returns_only_full_attention_layers(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(
                    layer_types=[
                        "linear_attention",
                        "full_attention",
                        "linear_attention",
                        "full_attention",
                    ]
                )
            )
        )
        self.assertEqual(discover_full_attention_layers(model), [1, 3])

    def test_hybrid_model_without_full_attention_fails_fast(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(layer_types=["linear_attention"])
            )
        )
        with self.assertRaisesRegex(ValueError, "full_attention"):
            discover_full_attention_layers(model)

    def test_non_hybrid_model_uses_real_layer_count(self):
        layers = [object(), object(), object()]
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(text_config=types.SimpleNamespace()),
            model=types.SimpleNamespace(layers=layers),
        )
        self.assertEqual(discover_full_attention_layers(model), [0, 1, 2])


class GridTest(unittest.TestCase):
    def test_uses_merged_processor_grids_for_multiple_images(self):
        self.assertEqual(
            resolve_visual_grids([[1, 8, 12], [1, 4, 8]], 2, 32, 2),
            ((4, 6), (2, 4)),
        )

    def test_grid_mismatch_reports_expected_and_actual(self):
        with self.assertRaisesRegex(ValueError, "expected 32.*actual 31"):
            resolve_visual_grids([[1, 8, 12], [1, 4, 8]], 2, 31, 2)
        with self.assertRaisesRegex(ValueError, "metadata"):
            resolve_visual_grids(None, 2, 24, 1)
        with self.assertRaisesRegex(ValueError, "2 images.*1 grid"):
            resolve_visual_grids([[1, 8, 12]], 2, 24, 2)

    def test_rejects_wrong_per_image_patch_partition_even_when_total_matches(self):
        visual_positions = np.array([1, 2, 4, 5, 6, 7])
        validate_visual_token_groups(visual_positions, ((1, 2), (2, 2)))
        with self.assertRaisesRegex(ValueError, "Image 1.*expected 4.*actual 2"):
            validate_visual_token_groups(visual_positions, ((2, 2), (1, 2)))


class HeadSelectionTest(unittest.TestCase):
    def test_mean_is_computed_in_float32(self):
        values = np.array([[[1.0, 2.0]], [[3.0, 6.0]]], dtype=np.float16)
        result = select_attention_head(values, "mean")
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[2.0, 4.0]], rtol=1e-5, atol=1e-7)


class FakeHookHandle:
    def __init__(self, hooks, hook):
        self.hooks = hooks
        self.hook = hook

    def remove(self):
        self.hooks.remove(self.hook)


class FakeAttentionModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook, with_kwargs=False):
        self.hooks.append(hook)
        return FakeHookHandle(self.hooks, hook)

    def emit(self, weights):
        output = (torch.zeros(1), weights)
        for hook in list(self.hooks):
            output = hook(self, (), {}, output)
        return output

    def emit_output(self, output):
        for hook in list(self.hooks):
            output = hook(self, (), {}, output)
        return output


class CaptureTest(unittest.TestCase):
    def _model(self):
        modules = [FakeAttentionModule(), FakeAttentionModule()]
        layers = [types.SimpleNamespace(self_attn=module) for module in modules]
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
        return model, modules

    def test_capture_slices_and_strips_full_attention(self):
        model, modules = self._model()
        weights = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape(1, 2, 6, 6)

        with AttentionCapture(
            model,
            layer_indices=[1],
            query_positions=np.array([3, 4]),
            visual_key_positions=np.array([1, 2]),
            context_key_positions=np.array([0, 3, 4, 5]),
        ) as capture:
            output = modules[1].emit(weights)
            self.assertIsNone(output[1])

        self.assertEqual(modules[1].hooks, [])
        layer = capture.layers[1]
        self.assertEqual(layer.visual.shape, (2, 2, 2))
        self.assertEqual(layer.context.shape, (2, 2, 4))
        self.assertEqual(layer.visual.dtype, np.float16)
        expected = weights[0, :, [3, 4]][:, :, [1, 2]].numpy().astype(np.float16)
        np.testing.assert_array_equal(layer.visual, expected)

    def test_hooks_removed_when_capture_body_raises(self):
        model, modules = self._model()
        with self.assertRaisesRegex(RuntimeError, "forward failed"), AttentionCapture(
            model,
            [0, 1],
            np.array([1]),
            np.array([0]),
            np.array([1]),
        ):
            raise RuntimeError("forward failed")
        self.assertEqual(modules[0].hooks, [])
        self.assertEqual(modules[1].hooks, [])

    def test_invalid_attention_output_fails_and_removes_hook(self):
        model, modules = self._model()
        with self.assertRaisesRegex(
            RuntimeError, "unsupported attention output"
        ), AttentionCapture(
            model,
            [0],
            np.array([1]),
            np.array([0]),
            np.array([1]),
        ):
            modules[0].emit_output(torch.zeros(1))
        self.assertEqual(modules[0].hooks, [])

    def test_second_capture_does_not_duplicate_hooks_or_results(self):
        model, modules = self._model()
        first_weights = torch.ones((1, 1, 3, 3))
        second_weights = torch.full((1, 1, 3, 3), 2.0)
        captures = []
        for weights in (first_weights, second_weights):
            with AttentionCapture(
                model,
                [0],
                np.array([1]),
                np.array([0]),
                np.array([1, 2]),
            ) as capture:
                modules[0].emit(weights)
            captures.append(capture)
            self.assertEqual(modules[0].hooks, [])
        self.assertEqual(float(captures[0].layers[0].visual[0, 0, 0]), 1.0)
        self.assertEqual(float(captures[1].layers[0].visual[0, 0, 0]), 2.0)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = types.SimpleNamespace(patch_size=2)
        self.messages = None
        self.call_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "templated"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        self.processor_images = kwargs["images"]
        return {
            "input_ids": torch.tensor([[10, 99, 99, 11, 99, 99, 99, 99, 11]]),
            "attention_mask": torch.ones((1, 9), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 2, 4], [1, 4, 4]]),
        }


class FakeModel:
    def __init__(self):
        self.modules = [FakeAttentionModule(), FakeAttentionModule()]
        layers = [types.SimpleNamespace(self_attn=module) for module in self.modules]
        self.model = types.SimpleNamespace(layers=layers)
        self.config = types.SimpleNamespace(
            model_type="fixture",
            image_token_id=99,
            text_config=types.SimpleNamespace(
                layer_types=["linear_attention", "full_attention"]
            ),
            vision_config=types.SimpleNamespace(
                patch_size=2,
                spatial_merge_size=2,
            ),
        )
        self.generate_calls = 0
        self.forward_calls = 0

    def generate(self, input_ids, **kwargs):
        self.generate_calls += 1
        generated = torch.tensor([[20, 12]], device=input_ids.device)
        return torch.cat((input_ids, generated), dim=1)

    def __call__(self, input_ids, **kwargs):
        self.forward_calls += 1
        length = input_ids.shape[1]
        weights = torch.arange(2 * length * length, dtype=torch.float32).reshape(
            1, 2, length, length
        )
        self.modules[1].emit(weights)
        return types.SimpleNamespace()


class ExtractionIntegrationTest(unittest.TestCase):
    def test_builds_compact_multi_image_session_from_full_attention_layers_only(self):
        model = FakeModel()
        processor = FakeProcessor()
        source_images = [
            Image.new("L", (4, 4), 64),
            Image.new("RGB", (8, 4), "blue"),
        ]
        processed_images = [
            Image.new("RGB", (8, 4)),
            Image.new("RGB", (8, 8)),
        ]
        vision_call = {}

        def process_vision_info(messages, **kwargs):
            vision_call.update(kwargs)
            return processed_images, None

        vision_module = types.SimpleNamespace(
            process_vision_info=process_vision_info
        )

        with mock.patch.dict("sys.modules", {"qwen_vl_utils": vision_module}):
            session = extract_attention(
                model,
                processor,
                source_images,
                user_prompt="describe",
                system_prompt="",
                max_new_tokens=2,
                device="cpu",
                resize_sizes=((6, 5), (7, 9)),
            )

        self.assertEqual([message["role"] for message in processor.messages], ["user"])
        self.assertEqual(model.generate_calls, 1)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(list(session.layers), [1])
        self.assertEqual(session.generated_positions.tolist(), [9, 10])
        self.assertEqual(session.selectable_positions.tolist(), [0, 3, 8, 9, 10])
        self.assertEqual(session.query_positions.tolist(), [0, 3, 8, 9, 10])
        self.assertEqual(session.layers[1].visual.shape, (2, 5, 6))
        self.assertEqual(session.layers[1].context.shape, (2, 5, 5))
        self.assertEqual(session.visual_grid_hws, ((1, 2), (2, 2)))
        self.assertEqual([image.mode for image in session.images], ["RGB", "RGB"])
        self.assertEqual(session.model_input_sizes, ((8, 4), (8, 8)))
        self.assertEqual(len(processor.messages[-1]["content"]), 3)
        self.assertEqual(len(processor.processor_images), 2)
        image_contents = processor.messages[0]["content"][:2]
        self.assertEqual(
            [
                (content["resized_width"], content["resized_height"])
                for content in image_contents
            ],
            [(6, 5), (7, 9)],
        )
        self.assertEqual(vision_call["image_patch_size"], 2)
        self.assertFalse(processor.call_kwargs["do_resize"])
        self.assertEqual(model.modules[0].hooks, [])
        self.assertEqual(model.modules[1].hooks, [])

    def test_supports_text_only_input_without_resize_or_vision_preprocessing(self):
        class TextOnlyProcessor(FakeProcessor):
            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }

        model = FakeModel()
        processor = TextOnlyProcessor()
        session = extract_attention(
            model,
            processor,
            (),
            user_prompt="describe",
            max_new_tokens=2,
            device="cpu",
            resize_sizes=(),
        )

        self.assertEqual(processor.messages[0]["content"], [{"type": "text", "text": "describe"}])
        self.assertNotIn("images", processor.call_kwargs)
        self.assertNotIn("do_resize", processor.call_kwargs)
        self.assertEqual(session.images, ())
        self.assertEqual(session.model_input_sizes, ())
        self.assertEqual(session.visual_grid_hws, ())
        self.assertEqual(session.visual_key_positions.tolist(), [])
        self.assertEqual(session.selectable_positions.tolist(), [0, 1, 2, 3])
        self.assertEqual(session.layers[1].visual.shape, (2, 4, 0))
        self.assertEqual(session.layers[1].context.shape, (2, 4, 4))

    def test_supports_image_only_input(self):
        class OneImageProcessor(FakeProcessor):
            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                self.processor_images = kwargs["images"]
                return {
                    "input_ids": torch.tensor([[10, 99, 99, 11]]),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long),
                    "image_grid_thw": torch.tensor([[1, 2, 4]]),
                }

        model = FakeModel()
        processor = OneImageProcessor()
        processed_image = Image.new("RGB", (8, 4))
        vision_module = types.SimpleNamespace(
            process_vision_info=lambda messages, **kwargs: ([processed_image], None)
        )

        with mock.patch.dict("sys.modules", {"qwen_vl_utils": vision_module}):
            session = extract_attention(
                model,
                processor,
                (Image.new("RGB", (4, 4)),),
                user_prompt="  ",
                max_new_tokens=2,
                device="cpu",
                resize_sizes=((4, 4),),
            )

        self.assertEqual(len(processor.messages[0]["content"]), 1)
        self.assertEqual(processor.messages[0]["content"][0]["type"], "image")
        self.assertEqual(session.generated_positions.tolist(), [4, 5])


@unittest.skipUnless(
    os.getenv("QWEN35_MODEL_PATH") and torch.cuda.is_available(),
    "QWEN35_MODEL_PATH and CUDA are required for the live Qwen3.5 smoke test",
)
class Qwen35SmokeTest(unittest.TestCase):
    def test_live_full_attention_capture(self):
        model_path = os.environ["QWEN35_MODEL_PATH"]
        model, processor = load_model(model_path, device="cuda:0")
        torch.cuda.reset_peak_memory_stats()

        session = extract_attention(
            model,
            processor,
            [
                Image.new("RGB", (64, 64), "white"),
                Image.new("RGB", (64, 64), "black"),
            ],
            user_prompt="Describe this image in one word.",
            max_new_tokens=2,
            device="cuda:0",
            resize_sizes=((64, 64), (96, 64)),
        )

        expected_layers = discover_full_attention_layers(model)
        self.assertEqual(list(session.layers), expected_layers)
        self.assertTrue(any(token.is_image for token in session.tokens))
        self.assertTrue(any(token.is_special and not token.is_image for token in session.tokens))
        self.assertEqual(len(session.images), 2)
        self.assertEqual(len(session.visual_grid_hws), 2)
        np.testing.assert_array_equal(
            session.query_positions,
            session.selectable_positions,
        )
        for layer in session.layers.values():
            self.assertEqual(layer.visual.ndim, 3)
            self.assertEqual(layer.context.ndim, 3)
            self.assertEqual(layer.visual.shape[1], len(session.selectable_positions))
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Qwen3.5 smoke peak CUDA memory: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    unittest.main()
