import html
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from vlm_attention_viz.app import (
    SessionCache,
    append_image_resize_specs,
    build_selector_state,
    load_server_image,
    render_cached_session,
    render_workbench_selection,
    run_inference,
    token_ribbon_html,
)


def _token(position, piece, segment="prompt", *, image=False, special=False):
    return SimpleNamespace(
        absolute_position=position,
        token_id=1000 + position,
        raw_piece=piece,
        decoded_preview=piece,
        segment=segment,
        is_special=special,
        is_image=image,
    )


def _session(tag="session"):
    return SimpleNamespace(
        tag=tag,
        tokens=(
            _token(0, "<|im_start|>", special=True),
            _token(1, "assistant"),
            _token(2, "<|image_pad|>", image=True, special=True),
            _token(3, "\n", special=True),
            _token(4, "same", "generated"),
            _token(5, "same", "generated"),
            _token(6, "<|eos|>", "generated", special=True),
        ),
        selectable_positions=np.array([0, 1, 3, 4, 5, 6]),
        generated_positions=np.array([4, 5, 6]),
        query_positions=np.array([0, 1, 3, 4, 5, 6]),
        images=(Image.new("RGB", (1, 1), "white"), Image.new("RGB", (1, 1), "black")),
        visual_grid_hws=((1, 1), (1, 1)),
        model_input_sizes=((32, 32), (32, 32)),
        layers={
            2: SimpleNamespace(visual=np.zeros((2, 6, 2))),
            7: SimpleNamespace(visual=np.zeros((3, 6, 2))),
        },
    )


def _resize_specs(*sizes):
    return [{"width": width, "height": height} for width, height in sizes]


class ResizeSpecTest(unittest.TestCase):
    def test_append_preserves_existing_specs_without_inspecting_old_pixels(self):
        first = Image.new("RGB", (10, 20), "red")
        specs = append_image_resize_specs([first])
        specs[0]["width"] = 101
        specs[0]["height"] = 102

        reencoded_first = Image.new("RGB", (10, 20), "pink")
        second = Image.new("RGB", (30, 40), "blue")
        appended = append_image_resize_specs([reencoded_first, second], specs)

        self.assertEqual(
            [(spec["width"], spec["height"]) for spec in appended],
            [(101, 102), (30, 40)],
        )
        self.assertEqual(appended[0]["id"], specs[0]["id"])
        self.assertNotEqual(appended[0]["id"], appended[1]["id"])

    def test_deletion_requires_the_indexed_delete_event(self):
        specs = append_image_resize_specs(
            [Image.new("RGB", (10, 20)), Image.new("RGB", (30, 40))]
        )
        with self.assertRaisesRegex(ValueError, "delete event"):
            append_image_resize_specs([Image.new("RGB", (10, 20))], specs)


class SessionCacheTest(unittest.TestCase):
    def test_uses_opaque_ids_and_evicts_oldest_entry(self):
        cache = SessionCache(max_entries=2)
        first_id = cache.put(_session("first"))
        second_id = cache.put(_session("second"))
        third_id = cache.put(_session("third"))

        self.assertNotEqual(first_id, second_id)
        self.assertNotIn("first", first_id)
        with self.assertRaisesRegex(KeyError, "expired"):
            cache.get(first_id)
        self.assertEqual(cache.get(second_id).tag, "second")
        self.assertEqual(cache.get(third_id).tag, "third")

    def test_browser_sessions_are_isolated(self):
        cache = SessionCache(max_entries=2)
        left = cache.put(_session("left"))
        right = cache.put(_session("right"))
        self.assertEqual(cache.get(left).tag, "left")
        self.assertEqual(cache.get(right).tag, "right")


class CallbackTest(unittest.TestCase):
    def test_image_or_text_is_sufficient_but_both_missing_is_rejected(self):
        calls = []

        def extractor(**kwargs):
            calls.append(kwargs)
            return _session()

        cache = SessionCache()
        with self.assertRaisesRegex(ValueError, "image or non-empty user prompt"):
            run_inference(
                cache,
                extractor,
                object(),
                object(),
                None,
                "",
                "  ",
                32,
                "cpu",
                [],
            )
        with self.assertRaisesRegex(ValueError, "Resize settings"):
            run_inference(
                cache,
                extractor,
                object(),
                object(),
                [
                    (Image.new("RGB", (2, 2)), None),
                    (Image.new("RGB", (3, 3)), None),
                ],
                "",
                "describe",
                32,
                "cpu",
                _resize_specs((2, 2)),
            )
        self.assertEqual(calls, [])

        run_inference(
            cache, extractor, object(), object(), None, "", "describe", 32, "cpu", []
        )
        run_inference(
            cache,
            extractor,
            object(),
            object(),
            [(Image.new("RGB", (2, 2)), None)],
            "",
            "  ",
            32,
            "cpu",
            _resize_specs((2, 2)),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["images"], ())
        self.assertEqual(calls[0]["resize_sizes"], ())

    def test_success_is_cached_and_failure_is_not(self):
        cache = SessionCache()
        calls = []

        def extractor(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary failure")
            return _session()

        images = [(Image.new("RGB", (2, 2)), None), (Image.new("RGB", (3, 2)), "second")]
        with self.assertRaisesRegex(RuntimeError, "temporary"):
            run_inference(
                cache,
                extractor,
                "model",
                "processor",
                images,
                "sys",
                "user",
                12,
                "cpu",
                resize_specs=_resize_specs((2, 2), (3, 4)),
            )
        self.assertEqual(len(cache), 0)

        session_id, selector = run_inference(
            cache,
            extractor,
            "model",
            "processor",
            images,
            "sys",
            "user",
            12,
            "cpu",
            resize_specs=_resize_specs((2, 2), (3, 4)),
        )
        self.assertEqual(len(cache), 1)
        self.assertIs(cache.get(session_id), selector.session)
        self.assertEqual(calls[-1]["system_prompt"], "sys")
        self.assertEqual(calls[-1]["user_prompt"], "user")
        self.assertEqual([image.size for image in calls[-1]["images"]], [(2, 2), (3, 2)])

    def test_per_image_dimensions_are_forwarded_without_aspect_fit(self):
        calls = []

        def extractor(**kwargs):
            calls.append(kwargs)
            return _session()

        run_inference(
            SessionCache(),
            extractor,
            object(),
            object(),
            [
                (Image.new("RGB", (400, 200)), None),
                (Image.new("RGB", (300, 500)), None),
            ],
            "",
            "describe",
            8,
            "cpu",
            resize_specs=_resize_specs((100, 75), (240, 360)),
        )

        self.assertEqual(calls[0]["resize_sizes"], ((100, 75), (240, 360)))

    def test_render_only_changes_never_invoke_inference(self):
        cache = SessionCache()
        inference_calls = []
        render_calls = []

        def extractor(**kwargs):
            inference_calls.append(kwargs)
            return _session()

        def renderer(session, generated_position, layer, head, image_index, opacity):
            render_calls.append(
                (session.tag, generated_position, layer, head, image_index, opacity)
            )
            return {"selection": render_calls[-1]}

        session_id, _ = run_inference(
            cache,
            extractor,
            "model",
            "processor",
            [(Image.new("RGB", (2, 2)), None)],
            "",
            "user",
            8,
            "cpu",
            resize_specs=_resize_specs((2, 2)),
        )
        for selection in (
            (4, 2, "Mean", 0, 0.3),
            (5, 7, 1, 1, 0.8),
            (6, 7, 2, 0, 1.0),
        ):
            render_cached_session(cache, session_id, *selection, renderer=renderer)

        self.assertEqual(len(inference_calls), 1)
        self.assertEqual(len(render_calls), 3)

    def test_workbench_render_keeps_only_causal_context_and_image_mass(self):
        session = _session()
        session.images = (
            Image.new("RGB", (2, 2), "white"),
            Image.new("RGB", (2, 2), "black"),
        )
        session.visual_grid_hws = ((1, 1), (1, 1))
        session.context_key_positions = np.array([0, 1, 3, 4, 5, 6])
        session.causal_context_mask = np.array(
            [
                [False, False, False, False, False, False],
                [True, False, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
            ]
        )
        session.layers = {
            2: SimpleNamespace(
                visual=np.array(
                    [
                        [[0.1, 0.6], [0.1, 0.6], [0.1, 0.6], [0.2, 0.7], [0.3, 0.8], [0.4, 0.9]],
                        [[0.1, 0.8], [0.1, 0.8], [0.1, 0.8], [0.4, 0.9], [0.5, 1.0], [0.6, 1.1]],
                    ]
                ),
                context=np.ones((2, 6, 6), dtype=np.float32) * 0.1,
            )
        }

        rendered = render_workbench_selection(session, 4, 2, "Mean", 1, 0.5)

        self.assertIn('data-position="0"', rendered.context_html)
        self.assertIn('data-position="3"', rendered.context_html)
        self.assertNotIn('data-position="4"', rendered.context_html)
        self.assertNotIn('data-position="5"', rendered.context_html)
        self.assertAlmostEqual(rendered.mass_summary["image_1"], 0.3)
        self.assertAlmostEqual(rendered.mass_summary["image_2"], 0.8)
        self.assertAlmostEqual(rendered.mass_summary["image_total"], 1.1)


class ServerImageTest(unittest.TestCase):
    def test_loads_relative_path_from_first_allowed_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (17, 11), "red").save(image_path)

            loaded = load_server_image("sample.png", [root])

        self.assertEqual(loaded.size, (17, 11))
        self.assertEqual(loaded.getpixel((0, 0)), (255, 0, 0))

    def test_rejects_paths_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "allowed"
            root.mkdir()
            outside = parent / "outside.png"
            Image.new("RGB", (2, 2)).save(outside)

            with self.assertRaisesRegex(ValueError, "allowed image roots"):
                load_server_image(outside, [root])

    def test_rejects_missing_and_non_image_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_server_image("missing.png", [root])
            with self.assertRaisesRegex(ValueError, "valid image"):
                load_server_image("notes.txt", [root])


class PresentationTest(unittest.TestCase):
    def test_ribbon_hides_only_images_and_preserves_absolute_positions(self):
        markup = token_ribbon_html(_session(), selected_position=5)
        self.assertNotIn("image_pad", markup)
        self.assertIn("&lt;|im_start|&gt;", markup)
        self.assertIn("\\n", markup)
        self.assertEqual(markup.count(">same<"), 2)
        self.assertEqual(markup.count('<button type="button"'), 6)
        self.assertIn('aria-label="Select text token at position 0"', markup)
        self.assertIn('data-position="4"', markup)
        self.assertIn('data-position="5"', markup)
        self.assertIn('aria-current="true"', markup)
        self.assertIn('aria-pressed="true"', markup)
        self.assertIn("Token ID: 1005", html.unescape(markup))
        self.assertIn("Segment: generated", html.unescape(markup))

    def test_visible_labels_use_contextual_decode_instead_of_raw_bpe_piece(self):
        session = _session()
        session.tokens[4].raw_piece = "\u0120same"
        session.tokens[4].decoded_preview = " same"

        markup = token_ribbon_html(session, selected_position=4)
        decoded_markup = html.unescape(markup)
        self.assertIn("> same</button>", decoded_markup)
        self.assertNotIn("\u2420", decoded_markup)
        self.assertNotIn(">\u0120same</button>", decoded_markup)

    def test_selector_defaults_and_layer_specific_heads(self):
        selector = build_selector_state(_session())
        self.assertEqual(selector.selectable_positions, (0, 1, 3, 4, 5, 6))
        self.assertEqual(selector.layers, (2, 7))
        self.assertEqual(selector.default_position, 4)
        self.assertEqual(selector.default_layer, 2)
        self.assertEqual(selector.image_indices, (0, 1))
        self.assertEqual(selector.default_image_index, 0)
        self.assertEqual(selector.heads_for(2), ("Mean", 0, 1))
        self.assertEqual(selector.heads_for(7), ("Mean", 0, 1, 2))

    def test_text_only_selector_has_no_default_image(self):
        session = _session()
        session.images = ()
        session.visual_grid_hws = ()
        session.model_input_sizes = ()
        session.layers = {
            2: SimpleNamespace(visual=np.empty((2, 6, 0))),
        }

        selector = build_selector_state(session)

        self.assertEqual(selector.image_indices, ())
        self.assertIsNone(selector.default_image_index)


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is not installed")
class GradioBuildTest(unittest.TestCase):
    def test_builds_blocks_without_running_inference(self):
        import gradio as gr

        extractor = Mock()
        fake_render_module = SimpleNamespace(render_selection=Mock())
        with patch.dict(sys.modules, {"vlm_attention_viz.render": fake_render_module}):
            demo = __import__("vlm_attention_viz.app", fromlist=["build_app"]).build_app(
                object(),
                object(),
                device="cpu",
                extractor=extractor,
            )

        self.assertIsInstance(demo, gr.Blocks)
        self.assertGreaterEqual(len(demo.config["dependencies"]), 4)
        run_functions = [
            block_function
            for block_function in demo.fns.values()
            if getattr(block_function.fn, "__name__", "") == "on_run"
        ]
        self.assertEqual(run_functions[0].concurrency_limit, 1)
        dependencies = demo.config["dependencies"]
        self.assertTrue(dependencies[0]["queue"])
        self.assertTrue(all(not dependency["queue"] for dependency in dependencies[1:]))
        self.assertIn("#token-ribbon button.token-selectable", demo.config["js"])
        labels = {
            component.get("props", {}).get("label")
            for component in demo.config["components"]
        }
        mass_component = next(
            component
            for component in demo.config["components"]
            if component.get("props", {}).get("label") == "Raw attention mass"
        )
        self.assertEqual(mass_component["type"], "code")
        self.assertEqual(mass_component["props"]["language"], "json")
        self.assertNotIn("Generated token position", labels)
        self.assertNotIn("Model input size", labels)
        self.assertIn("Resize width", labels)
        self.assertIn("Resize height", labels)
        self.assertIn("Server image path", labels)
        server_path = next(
            component
            for component in demo.blocks.values()
            if getattr(component, "label", None) == "Server image path"
        )
        self.assertEqual(server_path.lines, 1)
        self.assertEqual(server_path.max_lines, 1)
        components_by_id = {
            component.get("props", {}).get("elem_id"): component.get("props", {})
            for component in demo.config["components"]
        }
        self.assertIsNone(components_by_id["attention-image-input"].get("height"))
        self.assertEqual(components_by_id["system-prompt-input"]["lines"], 1)
        self.assertEqual(components_by_id["system-prompt-input"]["max_lines"], 1)
        self.assertEqual(components_by_id["user-prompt-input"]["lines"], 4)
        self.assertEqual(components_by_id["user-prompt-input"]["max_lines"], 4)
        self.assertIn("#attention-image-input", demo.css)
        self.assertIn("#image-input-column", demo.css)
        self.assertIn("#server-image-path-row", demo.css)
        self.assertIn("#server-image-path-input", demo.css)
        self.assertIn("#load-server-image-button", demo.css)
        self.assertIn("#prompt-input-column", demo.css)
        self.assertIn("#system-prompt-row", demo.css)
        self.assertIn("#user-prompt-row", demo.css)
        self.assertIn("#user-prompt-row > .form", demo.css)
        self.assertNotIn("#prompt-input-column > .form", demo.css)
        self.assertIn("background:transparent !important", demo.css)
        self.assertIn("border:0 !important", demo.css)
        self.assertIn("box-shadow:none !important", demo.css)
        self.assertIn("#system-prompt-input { padding-bottom:3px", demo.css)
        self.assertIn("padding-top:3px !important", demo.css)
        self.assertIn("#resize-input-column", demo.css)
        self.assertIn("#execution-input-column", demo.css)
        self.assertIn("#resize-input-row", demo.css)
        self.assertIn("#secondary-input-row { height:auto", demo.css)
        self.assertIn("align-self:stretch !important", demo.css)
        self.assertIn("gap:0 !important", demo.css)
        self.assertIn("display:grid", demo.css)
        self.assertIn("grid-template-rows:auto auto", demo.css)
        self.assertIn("flex-wrap:nowrap", demo.css)
        self.assertIn("#execution-input-column > .form", demo.css)
        self.assertIn("#max-new-tokens { width:100%", demo.css)
        self.assertIn("#run-button { width:100%", demo.css)
        self.assertIn("width:90vw !important", demo.css)
        self.assertIn("margin-inline:auto !important", demo.css)
        self.assertIn("height:32vh !important", demo.css)
        self.assertNotIn("height:14vh !important", demo.css)
        self.assertIn("flex:0 0 auto !important", demo.css)
        self.assertNotIn("max-width: 1440px", demo.css)
        self.assertIn("#system-prompt-input input", demo.css)
        self.assertIn("#user-prompt-input > label", demo.css)
        self.assertIn("#user-prompt-input .input-container", demo.css)
        self.assertIn("#user-prompt-input textarea", demo.css)
        self.assertIn("align-items:stretch !important", demo.css)
        self.assertIn("resize:none", demo.css)
        self.assertIn("Images", labels)
        self.assertIn("Overlay image", labels)

        self.assertTrue(
            any(
                getattr(block_function.fn, "__name__", "") == "on_token_click"
                for block_function in demo.fns.values()
            )
        )
        upload_function = next(
            block_function
            for block_function in demo.fns.values()
            if getattr(block_function.fn, "__name__", "") == "on_image_upload"
        )
        first = Image.new("RGB", (320, 240), "red")
        synced, selected_id, width_update, height_update = upload_function.fn(
            [first],
            [],
        )
        self.assertEqual(selected_id, synced[0]["id"])
        self.assertEqual(width_update["value"], 320)
        self.assertEqual(height_update["value"], 240)
        self.assertEqual(width_update["label"], "Resize width · Image 1")
        self.assertEqual(height_update["label"], "Resize height · Image 1")

        resize_width_function = next(
            block_function
            for block_function in demo.fns.values()
            if getattr(block_function.fn, "__name__", "") == "on_resize_width"
        )
        updated_specs = resize_width_function.fn(123, synced, selected_id)
        reencoded_first = Image.new("RGB", (320, 240), "pink")
        second = Image.new("RGB", (640, 360), "blue")
        appended, selected_id, width_update, height_update = upload_function.fn(
            [reencoded_first, second],
            updated_specs,
        )
        self.assertEqual(appended[0]["width"], 123)
        self.assertEqual(appended[1]["width"], 640)
        self.assertEqual(selected_id, appended[1]["id"])
        self.assertEqual(width_update["label"], "Resize width · Image 2")

        select_function = next(
            block_function
            for block_function in demo.fns.values()
            if getattr(block_function.fn, "__name__", "") == "on_image_select"
        )
        selected_id, selected_width, selected_height = select_function.fn(
            appended,
            SimpleNamespace(index=0),
        )
        self.assertEqual(selected_id, appended[0]["id"])
        self.assertEqual(selected_width["value"], 123)
        self.assertEqual(selected_width["label"], "Resize width · Image 1")
        self.assertEqual(selected_height["label"], "Resize height · Image 1")

        delete_function = next(
            block_function
            for block_function in demo.fns.values()
            if getattr(block_function.fn, "__name__", "") == "on_image_delete"
        )
        remaining, selected_id, selected_width, selected_height = delete_function.fn(
            appended,
            selected_id,
            SimpleNamespace(_data={"index": 1}),
        )
        self.assertEqual(len(remaining), 1)
        self.assertEqual(selected_id, remaining[0]["id"])
        self.assertEqual(selected_width["value"], 123)
        self.assertEqual(selected_width["label"], "Resize width · Image 1")

        remaining, selected_id, selected_width, selected_height = delete_function.fn(
            appended,
            appended[0]["id"],
            SimpleNamespace(_data={"index": 0}),
        )
        self.assertEqual(len(remaining), 1)
        self.assertEqual(selected_id, remaining[0]["id"])
        self.assertEqual(selected_width["value"], 640)
        self.assertEqual(selected_width["label"], "Resize width · Image 1")
        self.assertEqual(selected_height["value"], 360)
        self.assertEqual(
            sum(
                getattr(block_function.fn, "__name__", "")
                == "on_server_image_load"
                for block_function in demo.fns.values()
            ),
            2,
        )
        extractor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
