"""Pure NumPy/PIL rendering helpers for attention workbench slices.

The renderer deliberately knows nothing about models or Gradio.  A caller
selects a layer/query/head from an :class:`AttentionSession`, then passes the
resulting vectors here for normalization and display.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AttentionSlice:
    """The selected visual and context vectors, represented as float32."""

    visual: np.ndarray
    context: np.ndarray
    vmax: float
    visual_values: np.ndarray
    context_values: np.ndarray


@dataclass(frozen=True)
class RenderSelection:
    """Render outputs consumed by the thin Gradio callback."""

    overlay: Image.Image | None
    context_html: str
    mass_summary: dict[str, float]
    vmax: float
    visual: np.ndarray
    context: np.ndarray


def _finite_vector(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} attention must be a 1-D vector, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} attention contains NaN or Inf")
    return arr


def select_attention_slice(layer: Any, query_index: int, head: int | str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Select one query/head from a ``LayerAttention``-like object.

    ``head='mean'`` computes an immediate float32 mean and never stores a
    second mean tensor.  Integer heads are validated against both modalities.
    """
    visual = np.asarray(layer.visual)
    context = np.asarray(layer.context)
    if visual.ndim != 3 or context.ndim != 3:
        raise ValueError("layer visual/context arrays must have shape [heads, queries, keys]")
    if visual.shape[:2] != context.shape[:2]:
        raise ValueError("visual/context head and query dimensions must match")
    if not 0 <= query_index < visual.shape[1]:
        raise IndexError(f"query_index {query_index} outside [0, {visual.shape[1]})")
    if isinstance(head, str) and head.lower() == "mean":
        visual_values = np.asarray(visual[:, query_index, :], dtype=np.float32)
        context_values = np.asarray(context[:, query_index, :], dtype=np.float32)
        if not np.isfinite(visual_values).all() or not np.isfinite(context_values).all():
            raise ValueError("attention contains NaN or Inf")
        return (
            visual_values.mean(axis=0),
            context_values.mean(axis=0),
        )
    if not isinstance(head, (int, np.integer)):
        raise TypeError("head must be 'mean' or an integer head index")
    if not 0 <= int(head) < visual.shape[0]:
        raise IndexError(f"head {head} outside [0, {visual.shape[0]})")
    return (
        _finite_vector(visual[int(head), query_index], "visual"),
        _finite_vector(context[int(head), query_index], "context"),
    )


def normalize_slice(visual: np.ndarray | Iterable[float], context: np.ndarray | Iterable[float]) -> AttentionSlice:
    """Normalize both modalities using one ``[0, max(current slice)]`` scale."""
    visual_arr = _finite_vector(visual, "visual")
    context_arr = _finite_vector(context, "context")
    vmax = float(max(0.0, np.max(visual_arr, initial=0.0), np.max(context_arr, initial=0.0)))
    if vmax == 0.0:
        visual_values = np.zeros_like(visual_arr)
        context_values = np.zeros_like(context_arr)
    else:
        visual_values = np.clip(visual_arr / vmax, 0.0, 1.0)
        context_values = np.clip(context_arr / vmax, 0.0, 1.0)
    return AttentionSlice(visual_arr, context_arr, vmax, visual_values, context_values)


def attention_colors(visual: np.ndarray | Iterable[float], context: np.ndarray | Iterable[float]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return normalized visual/context values and the shared color maximum."""
    result = normalize_slice(visual, context)
    return result.visual_values, result.context_values, result.vmax


def _reshape_grid(values: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    gh, gw = (int(grid_hw[0]), int(grid_hw[1]))
    if gh <= 0 or gw <= 0:
        raise ValueError(f"grid dimensions must be positive, got {grid_hw}")
    if values.size != gh * gw:
        raise ValueError(f"visual attention/grid mismatch: expected {gh * gw}, got {values.size}")
    return values.reshape(gh, gw)


def render_heatmap(visual: np.ndarray | Iterable[float], grid_hw: tuple[int, int], size: tuple[int, int], *, vmax: float | None = None) -> Image.Image:
    """Render a normalized attention vector as an RGB heatmap at ``size``."""
    values = _finite_vector(visual, "visual")
    if vmax is None:
        vmax = float(max(0.0, np.max(values, initial=0.0)))
    if not np.isfinite(vmax) or vmax < 0:
        raise ValueError("vmax must be finite and non-negative")
    normalized = np.zeros_like(values) if vmax == 0 else np.clip(values / vmax, 0.0, 1.0)
    grid = _reshape_grid(normalized, grid_hw)
    # A compact blue-to-red map keeps the renderer dependency-free.
    if vmax == 0.0:
        rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
    else:
        red = np.asarray(np.round(grid * 255), dtype=np.uint8)
        blue = np.asarray(np.round((1.0 - grid) * 255), dtype=np.uint8)
        green = np.asarray(np.round((1.0 - np.abs(grid - 0.5) * 2.0) * 180), dtype=np.uint8)
        rgb = np.stack([red, green, blue], axis=-1)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return Image.fromarray(rgb, mode="RGB").resize((int(size[0]), int(size[1])), resampling)


def _visual_contrast(values: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = _finite_vector(values, "visual")
    if arr.size == 0:
        return np.zeros_like(arr)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


def overlay_attention(
    image: Image.Image,
    visual: np.ndarray | Iterable[float],
    grid_hw: tuple[int, int],
    alpha: float = 0.55,
) -> Image.Image:
    """Overlay a full blue-to-red map using an image-local attention scale."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    base = image.convert("RGB")
    contrast = _visual_contrast(visual)
    if float(alpha) == 0.0 or np.max(contrast, initial=0.0) == 0.0:
        return base
    heatmap = render_heatmap(contrast, grid_hw, base.size, vmax=1.0)
    return Image.blend(base, heatmap, float(alpha))


def split_visual_attention(
    visual: np.ndarray | Iterable[float],
    grid_hws: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, ...]:
    """Split concatenated visual attention using each image's merged patch grid."""
    values = _finite_vector(visual, "visual")
    parts = []
    offset = 0
    for image_index, grid_hw in enumerate(grid_hws):
        height, width = (int(grid_hw[0]), int(grid_hw[1]))
        if height <= 0 or width <= 0:
            raise ValueError(
                f"image {image_index + 1} grid dimensions must be positive, got {grid_hw}"
            )
        stop = offset + height * width
        parts.append(values[offset:stop])
        offset = stop
    if not parts and values.size:
        raise ValueError("visual attention exists without any visual grids")
    if offset != values.size:
        raise ValueError(
            f"visual attention/grid mismatch: grids require {offset} values, got {values.size}"
        )
    return tuple(parts)


def attention_masses(context: np.ndarray | Iterable[float], key_positions: Iterable[int], tokens: Iterable[Any]) -> dict[str, float]:
    """Sum raw context attention by prompt/generated and special/other groups."""
    values = _finite_vector(context, "context")
    positions = np.asarray(list(key_positions), dtype=np.int64)
    token_by_pos = {int(token.absolute_position): token for token in tokens}
    if positions.size != values.size:
        raise ValueError(f"context attention/key position mismatch: expected {values.size}, got {positions.size}")
    masses = {"prompt": 0.0, "generated_history": 0.0, "special_other": 0.0}
    for value, position in zip(values, positions):
        token = token_by_pos.get(int(position))
        if token is None:
            raise ValueError(f"missing token metadata for context key position {position}")
        segment = getattr(token, "segment", "")
        if bool(getattr(token, "is_special", False)):
            group = "special_other"
        elif segment == "prompt":
            group = "prompt"
        elif segment == "generated":
            group = "generated_history"
        else:
            group = "special_other"
        masses[group] += float(value)
    return masses


def _context_html(values: np.ndarray, vmax: float, key_positions: Iterable[int], tokens: Iterable[Any]) -> str:
    """Build a small escaped token stream; colors use the current slice scale."""
    by_position = {int(token.absolute_position): token for token in tokens}
    spans = []
    for value, position in zip(values, key_positions):
        token = by_position.get(int(position))
        if token is None:
            continue
        intensity = 0.0 if vmax == 0 else float(np.clip(value / vmax, 0, 1))
        piece = str(getattr(token, "raw_piece", "")) or "[empty]"
        decoded = str(getattr(token, "decoded_preview", ""))
        label = (decoded or "[byte]").replace("\n", "\\n").replace("\t", "\\t")
        title = escape(
            f"Raw piece: {piece}\nDecoded: {getattr(token, 'decoded_preview', '')}\n"
            f"Token ID: {getattr(token, 'token_id', '')}\nPosition: {position}",
            quote=True,
        )
        # Blue-to-red alpha communicates raw magnitude without changing key order.
        red = round(255 * intensity)
        blue = round(255 * (1 - intensity))
        spans.append(
            f'<span class="context-token" data-position="{int(position)}" title="{title}" '
            f'style="background:rgb({red},80,{blue});color:white">{escape(label)}</span>'
        )
    return '<div class="context-token-stream">' + "".join(spans) + "</div>"


def render_selection(
    session: Any,
    selected_position: int,
    layer: int,
    head: int | str = "Mean",
    image_index: int | None = None,
    opacity: float = 0.55,
) -> RenderSelection:
    """Render one query x full-layer x head selection from an existing session."""
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError("opacity must be between 0 and 1")
    positions = np.asarray(session.selectable_positions, dtype=np.int64)
    matches = np.flatnonzero(positions == int(selected_position))
    if matches.size != 1:
        raise ValueError(f"text position {selected_position} is not selectable")
    layer_obj = session.layers[int(layer)]
    visual, context = select_attention_slice(layer_obj, int(matches[0]), head)
    selected = normalize_slice(visual, context)
    overlay = None
    images = tuple(getattr(session, "images", ()))
    grid_hws = tuple(getattr(session, "visual_grid_hws", ()))
    if len(images) != len(grid_hws):
        raise ValueError("attention session image/grid counts do not match")
    if images:
        selected_image = 0 if image_index is None else int(image_index)
        if not 0 <= selected_image < len(images):
            raise IndexError(f"image {selected_image} outside [0, {len(images)})")
        visual_parts = split_visual_attention(selected.visual, grid_hws)
        overlay = overlay_attention(
            images[selected_image].convert("RGB"),
            visual_parts[selected_image],
            grid_hws[selected_image],
            float(opacity),
        )
    elif selected.visual.size:
        raise ValueError("visual attention exists without any source images")
    key_positions = getattr(session, "context_key_positions", range(len(context)))
    context_html = _context_html(
        selected.context, selected.vmax, key_positions, getattr(session, "tokens", ())
    )
    masses = attention_masses(selected.context, key_positions, getattr(session, "tokens", ()))
    return RenderSelection(overlay, context_html, masses, selected.vmax, selected.visual, selected.context)


__all__ = [
    "AttentionSlice",
    "RenderSelection",
    "attention_colors",
    "attention_masses",
    "normalize_slice",
    "overlay_attention",
    "render_heatmap",
    "render_selection",
    "select_attention_slice",
    "split_visual_attention",
]
