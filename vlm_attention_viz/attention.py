"""Model inference and compact attention-session construction."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class TokenInfo:
    absolute_position: int
    token_id: int
    raw_piece: str
    decoded_preview: str
    segment: Literal["prompt", "generated"]
    is_special: bool
    is_image: bool


@dataclass
class LayerAttention:
    visual: np.ndarray
    context: np.ndarray


@dataclass
class AttentionSession:
    tokens: tuple[TokenInfo, ...]
    selectable_positions: np.ndarray
    generated_positions: np.ndarray
    query_positions: np.ndarray
    visual_key_positions: np.ndarray
    context_key_positions: np.ndarray
    causal_context_mask: np.ndarray
    layers: dict[int, LayerAttention]
    visual_grid_hws: tuple[tuple[int, int], ...]
    images: tuple[Image.Image, ...]
    model_input_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _SessionMetadata:
    tokens: tuple[TokenInfo, ...]
    selectable_positions: np.ndarray
    generated_positions: np.ndarray
    query_positions: np.ndarray
    visual_key_positions: np.ndarray
    context_key_positions: np.ndarray
    causal_context_mask: np.ndarray


def load_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype=None,
):
    """Load a supported vision-language model using eager attention."""
    from transformers import AutoProcessor

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_class = _detect_model_class(model_path)
    model = model_class.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _detect_model_class(model_path: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", ""))
    path_hint = model_path.lower()
    if "qwen3_5_moe" in model_type or "qwen3.5-moe" in path_hint or "qwen3_5_moe" in path_hint:
        from transformers import Qwen3_5MoeForConditionalGeneration

        return Qwen3_5MoeForConditionalGeneration
    if "qwen3_5" in model_type or "qwen3.5" in path_hint or "qwen3_5" in path_hint:
        from transformers import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration
    if "qwen2_5_vl" in model_type or "qwen2.5" in path_hint or "qwen2_5" in path_hint:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    if "qwen2_vl" in model_type or "qwen2-vl" in path_hint:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    from transformers import AutoModelForVision2Seq

    return AutoModelForVision2Seq


def _language_layers(model):
    candidates = (
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "model", None),
    )
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise ValueError("Could not locate the language model layers on this model")


def _optional_language_layers(model):
    try:
        return _language_layers(model)
    except ValueError:
        return None


def discover_full_attention_layers(model) -> list[int]:
    """Return full-attention layer indices without architecture-size fallbacks."""
    config = model.config
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)

    if layer_types is not None:
        layer_types = list(layer_types)
        layers = _optional_language_layers(model)
        if layers is not None and len(layer_types) != len(layers):
            raise ValueError(
                "text_config.layer_types length does not match language layers: "
                f"expected {len(layers)}, actual {len(layer_types)}"
            )
        allowed = {"full_attention", "linear_attention"}
        unknown = sorted(set(layer_types) - allowed)
        if unknown:
            raise ValueError(f"Unsupported attention layer types: {unknown}")
        result = [
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        ]
        if not result:
            raise ValueError("text_config.layer_types contains no full_attention layers")
        return result

    model_type = str(getattr(config, "model_type", "")).lower()
    if "qwen3_5" in model_type:
        raise ValueError("Qwen3.5 requires non-empty text_config.layer_types metadata")
    layers = _language_layers(model)
    if not layers:
        raise ValueError("The language model has no layers")
    return list(range(len(layers)))


def build_session_metadata(
    tokenizer,
    full_ids: Sequence[int],
    prompt_length: int,
    image_token_id: int | None,
) -> _SessionMetadata:
    """Build the single absolute-position mapping used by the entire application."""
    ids = [int(token_id) for token_id in full_ids]
    if not 0 < prompt_length < len(ids):
        raise ValueError("At least one prompt token and one generated token are required")

    id_array = np.asarray(ids)
    if image_token_id is None:
        visual_positions = np.empty(0, dtype=np.int64)
    else:
        visual_positions = np.flatnonzero(id_array == int(image_token_id)).astype(np.int64)
        if visual_positions.size == 0:
            raise ValueError(f"No image token with id {image_token_id} exists in full_ids")

    raw_pieces = tokenizer.convert_ids_to_tokens(ids)
    if isinstance(raw_pieces, str):
        raw_pieces = [raw_pieces]
    if len(raw_pieces) != len(ids):
        raise ValueError("Tokenizer returned a different number of token pieces than full_ids")
    decoded_pieces = _decode_token_chunks(tokenizer, ids)
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())}
    tokens = tuple(
        TokenInfo(
            absolute_position=position,
            token_id=token_id,
            raw_piece=str(raw_pieces[position]),
            decoded_preview=decoded_pieces[position],
            segment="prompt" if position < prompt_length else "generated",
            is_special=token_id in special_ids,
            is_image=image_token_id is not None and token_id == image_token_id,
        )
        for position, token_id in enumerate(ids)
    )

    generated_positions = np.arange(prompt_length, len(ids), dtype=np.int64)
    selectable_mask = np.ones(len(ids), dtype=bool)
    selectable_mask[visual_positions] = False
    selectable_positions = np.flatnonzero(selectable_mask).astype(np.int64)
    query_positions = selectable_positions.copy()
    context_positions = selectable_positions.copy()
    causal_mask = context_positions[None, :] < selectable_positions[:, None]
    return _SessionMetadata(
        tokens=tokens,
        selectable_positions=selectable_positions,
        generated_positions=generated_positions,
        query_positions=query_positions,
        visual_key_positions=visual_positions,
        context_key_positions=context_positions,
        causal_context_mask=causal_mask,
    )


def _decode_token_chunks(tokenizer, ids: Sequence[int]) -> list[str]:
    """Decode token IDs incrementally so byte fragments retain sequence context."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        try:
            from tokenizers.decoders import DecodeStream
        except ImportError:
            pass
        else:
            stream = DecodeStream(skip_special_tokens=False)
            chunks = (stream.step(backend, token_id) for token_id in ids)
            return [str(chunk) if chunk is not None else "" for chunk in chunks]

    return [
        str(tokenizer.decode([token_id], skip_special_tokens=False))
        for token_id in ids
    ]


def resolve_visual_grids(
    image_grid_thw,
    spatial_merge_size: int,
    visual_token_count: int,
    image_count: int,
) -> tuple[tuple[int, int], ...]:
    """Validate processor metadata and return one merged patch grid per image."""
    if image_grid_thw is None:
        raise ValueError("Processor image_grid_thw metadata is required")
    if isinstance(image_grid_thw, torch.Tensor):
        grid = image_grid_thw.detach().cpu().numpy()
    else:
        grid = np.asarray(image_grid_thw)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"Image grid metadata must have shape (N, 3), actual {grid.shape}")
    if grid.shape[0] != int(image_count):
        raise ValueError(
            f"Expected {image_count} images but processor returned {grid.shape[0]} grid rows"
        )
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    merged_grids = []
    for image_index, row in enumerate(grid):
        temporal, height, width = (int(value) for value in row)
        if temporal != 1:
            raise ValueError(
                f"Image {image_index + 1} temporal grid must be 1, actual {temporal}"
            )
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                f"Image {image_index + 1} grid ({height}, {width}) is not divisible by "
                f"spatial_merge_size={spatial_merge_size}"
            )
        merged_grids.append(
            (height // spatial_merge_size, width // spatial_merge_size)
        )

    expected = sum(height * width for height, width in merged_grids)
    if expected != visual_token_count:
        raise ValueError(
            f"Merged patch count mismatch: expected {expected} from processor grids, "
            f"actual {visual_token_count} image tokens"
        )
    return tuple(merged_grids)


def validate_visual_token_groups(
    visual_positions: np.ndarray,
    grid_hws: Sequence[tuple[int, int]],
) -> None:
    """Ensure each image-pad run matches its processor grid in image order."""
    positions = np.asarray(visual_positions, dtype=np.int64)
    if not grid_hws:
        if positions.size:
            raise ValueError("Visual tokens exist without any image grids")
        return
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    groups = np.split(positions, boundaries)
    if len(groups) != len(grid_hws):
        raise ValueError(
            f"Expected {len(grid_hws)} image token groups, actual {len(groups)}"
        )
    for image_index, (group, grid_hw) in enumerate(zip(groups, grid_hws)):
        expected = int(grid_hw[0]) * int(grid_hw[1])
        if group.size != expected:
            raise ValueError(
                f"Image {image_index + 1} patch count mismatch: expected {expected}, "
                f"actual {group.size} image tokens"
            )


def select_attention_head(values: np.ndarray, head: int | str) -> np.ndarray:
    """Select one head or compute the unpersisted float32 arithmetic mean."""
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(
            f"Attention values must have shape [heads, queries, keys], actual {array.shape}"
        )
    if isinstance(head, str):
        if head.lower() != "mean":
            raise ValueError(f"Unknown head selection: {head}")
        return array.astype(np.float32).mean(axis=0)
    if not 0 <= head < array.shape[0]:
        raise IndexError(f"Head {head} is outside [0, {array.shape[0]})")
    return array[head].astype(np.float32)


class AttentionCapture:
    """Context-managed hooks that retain only compact query/key slices."""

    def __init__(
        self,
        model,
        layer_indices: Sequence[int],
        query_positions: np.ndarray,
        visual_key_positions: np.ndarray,
        context_key_positions: np.ndarray,
    ):
        self._model = model
        self._layer_indices = list(layer_indices)
        self._query_positions = np.asarray(query_positions, dtype=np.int64)
        self._visual_positions = np.asarray(visual_key_positions, dtype=np.int64)
        self._context_positions = np.asarray(context_key_positions, dtype=np.int64)
        self._handles = []
        self.layers: OrderedDict[int, LayerAttention] = OrderedDict()

    def __enter__(self):
        layers = _language_layers(self._model)
        try:
            for layer_index in self._layer_indices:
                if not 0 <= layer_index < len(layers):
                    raise IndexError(f"Layer {layer_index} is outside [0, {len(layers)})")
                self._handles.append(
                    layers[layer_index].self_attn.register_forward_hook(
                        self._make_hook(layer_index), with_kwargs=True
                    )
                )
        except Exception:
            self._remove_hooks()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()
        return False

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int):
        def hook(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(
                    f"Layer {layer_index} returned unsupported attention output; "
                    "expected a tuple/list with weights at index 1"
                )
            weights = output[1]
            if weights is None:
                raise RuntimeError(
                    f"Layer {layer_index} returned no attention weights; "
                    "eager attention is required"
                )
            if not isinstance(weights, torch.Tensor) or weights.ndim != 4 or weights.shape[0] != 1:
                shape = getattr(weights, "shape", None)
                raise RuntimeError(
                    f"Layer {layer_index} attention must have shape "
                    f"[1, heads, queries, keys], actual {shape}"
                )
            max_query = int(self._query_positions.max())
            key_groups = [
                positions
                for positions in (self._visual_positions, self._context_positions)
                if positions.size
            ]
            if not key_groups:
                raise RuntimeError("At least one attention key position is required")
            max_key = max(int(positions.max()) for positions in key_groups)
            if max_query >= weights.shape[2] or max_key >= weights.shape[3]:
                raise RuntimeError(
                    f"Layer {layer_index} attention shape {tuple(weights.shape)} cannot satisfy "
                    f"query position {max_query} and key position {max_key}"
                )

            query_index = torch.as_tensor(self._query_positions, device=weights.device)
            visual_index = torch.as_tensor(self._visual_positions, device=weights.device)
            context_index = torch.as_tensor(self._context_positions, device=weights.device)
            queried = weights[0].index_select(1, query_index)
            visual = queried.index_select(2, visual_index).detach().to("cpu", torch.float16).numpy()
            context = (
                queried.index_select(2, context_index)
                .detach()
                .to("cpu", torch.float16)
                .numpy()
            )
            self.layers[layer_index] = LayerAttention(visual=visual, context=context)

            stripped = list(output)
            stripped[1] = None
            return tuple(stripped) if isinstance(output, tuple) else stripped

        return hook


def extract_attention(
    model,
    processor,
    images: Sequence[Image.Image],
    user_prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 256,
    device: str = "cuda",
    resize_sizes: Sequence[tuple[int, int]] | None = None,
) -> AttentionSession:
    """Run generation from zero or more images and optional text."""
    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
        raise TypeError("images must be a sequence of PIL.Image.Image values")
    if any(not isinstance(image, Image.Image) for image in images):
        raise TypeError("Every input image must be a PIL.Image.Image")
    has_images = bool(images)
    has_text = bool(user_prompt and user_prompt.strip())
    if not has_images and not has_text:
        raise ValueError("At least one image or non-empty user prompt is required")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    images = tuple(
        image.convert("RGB") if image.mode != "RGB" else image.copy()
        for image in images
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = []
    patch_size = merge_size = 0
    if has_images:
        if resize_sizes is None or len(resize_sizes) != len(images):
            raise ValueError("resize_sizes must contain one size for every image")
        normalized_sizes = []
        for image_index, size in enumerate(resize_sizes):
            if len(size) != 2:
                raise ValueError(
                    f"Image {image_index + 1} resize size must contain width and height"
                )
            width, height = (int(dimension) for dimension in size)
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Image {image_index + 1} resize dimensions must be positive"
                )
            normalized_sizes.append((width, height))
        vision_config = getattr(model.config, "vision_config", None)
        image_processor = getattr(processor, "image_processor", None)
        patch_size = int(
            getattr(image_processor, "patch_size", 0)
            or getattr(vision_config, "patch_size", 0)
        )
        merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
        if patch_size <= 0 or merge_size <= 0:
            raise ValueError(
                "model.config.vision_config.patch_size and spatial_merge_size are required"
            )
        content.extend(
            {
                "type": "image",
                "image": image,
                "resized_width": width,
                "resized_height": height,
            }
            for image, (width, height) in zip(images, normalized_sizes)
        )
    if has_text:
        content.append({"type": "text", "text": user_prompt})

    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )
    template_kwargs = _thinking_kwargs(processor)
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    processor_kwargs = {"text": [text_input], "padding": True, "return_tensors": "pt"}
    model_input_sizes: tuple[tuple[int, int], ...] = ()
    if has_images:
        from qwen_vl_utils import process_vision_info

        image_inputs, _ = process_vision_info(messages, image_patch_size=patch_size)
        if len(image_inputs) != len(images) or any(
            not isinstance(image, Image.Image) for image in image_inputs
        ):
            raise ValueError("Vision preprocessing must return one PIL image per input image")
        model_input_sizes = tuple(
            tuple(int(dimension) for dimension in image.size) for image in image_inputs
        )
        spatial_factor = patch_size * merge_size
        incompatible_sizes = [
            size
            for size in model_input_sizes
            if any(dimension % spatial_factor for dimension in size)
        ]
        if incompatible_sizes:
            raise RuntimeError(
                "Vision preprocessing returned image sizes that are incompatible with "
                f"patch_size={patch_size} and spatial_merge_size={merge_size}: "
                f"{incompatible_sizes}. qwen-vl-utils>=0.0.14 is required."
            )
        processor_kwargs.update(images=image_inputs, do_resize=False)
    inputs = processor(**processor_kwargs)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Processor input_ids must have shape [1, prompt_tokens]")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if not isinstance(generated, torch.Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
        raise RuntimeError("model.generate must return token ids with shape [1, sequence]")
    prompt_length = input_ids.shape[1]
    includes_prompt = generated.shape[1] >= prompt_length and torch.equal(
        generated[0, :prompt_length], input_ids[0]
    )
    if includes_prompt:
        generated_ids = generated[:, prompt_length:]
    else:
        generated_ids = generated
    if generated_ids.shape[1] == 0:
        raise ValueError("Model returned no generated token")

    full_ids = torch.cat((input_ids, generated_ids.to(input_ids.device)), dim=1)
    image_token_id = _image_token_id(model, processor) if has_images else None
    metadata = build_session_metadata(
        processor.tokenizer,
        full_ids[0].tolist(),
        prompt_length,
        image_token_id,
    )
    visual_grid_hws = (
        resolve_visual_grids(
            inputs.get("image_grid_thw"),
            merge_size,
            len(metadata.visual_key_positions),
            len(images),
        )
        if has_images
        else ()
    )
    validate_visual_token_groups(metadata.visual_key_positions, visual_grid_hws)
    layer_indices = discover_full_attention_layers(model)

    full_inputs = dict(inputs)
    full_inputs["input_ids"] = full_ids
    full_inputs["attention_mask"] = torch.ones_like(full_ids, dtype=torch.long)
    full_inputs.pop("position_ids", None)
    full_inputs.pop("token_type_ids", None)
    with AttentionCapture(
        model,
        layer_indices,
        metadata.query_positions,
        metadata.visual_key_positions,
        metadata.context_key_positions,
    ) as capture, torch.no_grad():
        model(**full_inputs, output_attentions=True, use_cache=False)
    missing = sorted(set(layer_indices) - set(capture.layers))
    if missing:
        raise RuntimeError(f"No attention was captured for full-attention layers: {missing}")

    return AttentionSession(
        tokens=metadata.tokens,
        selectable_positions=metadata.selectable_positions,
        generated_positions=metadata.generated_positions,
        query_positions=metadata.query_positions,
        visual_key_positions=metadata.visual_key_positions,
        context_key_positions=metadata.context_key_positions,
        causal_context_mask=metadata.causal_context_mask,
        layers=dict(capture.layers),
        visual_grid_hws=visual_grid_hws,
        images=images,
        model_input_sizes=model_input_sizes,
    )


def _image_token_id(model, processor) -> int:
    candidates = (
        getattr(model.config, "image_token_id", None),
        getattr(getattr(model.config, "vision_config", None), "image_token_id", None),
        getattr(processor.tokenizer, "image_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError("The model/processor does not provide an image_token_id")


def _thinking_kwargs(processor) -> dict:
    try:
        processor.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return {}
    return {"enable_thinking": False}
