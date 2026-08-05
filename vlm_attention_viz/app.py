"""Gradio workbench and thin callbacks for attention sessions."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

TOKEN_RIBBON_JS = r"""
() => {
  if (window.__vlmAttentionTokenRibbonBound) return;
  window.__vlmAttentionTokenRibbonBound = true;
  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest("#token-ribbon button.token-selectable");
    if (!button) return;
    const bridge = document.querySelector(
      "#token-click-bridge textarea, #token-click-bridge input"
    );
    if (!bridge) return;
    const value = `${button.dataset.position}:${Date.now()}:${Math.random()}`;
    const prototype = bridge instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, "value").set.call(bridge, value);
    bridge.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
"""


class SessionCache:
    """Bounded process-local storage for attention sessions."""

    def __init__(self, max_entries: int = 8):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._sessions: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def put(self, session: Any) -> str:
        session_id = str(uuid4())
        with self._lock:
            self._sessions[session_id] = session
            while len(self._sessions) > self.max_entries:
                self._sessions.popitem(last=False)
        return session_id

    def get(self, session_id: str) -> Any:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError("Attention session is missing or expired; run inference again") from exc


@dataclass(frozen=True)
class SelectorState:
    """UI selector values derived only from an attention session."""

    session: Any
    selectable_positions: tuple[int, ...]
    layers: tuple[int, ...]
    heads_by_layer: dict[int, tuple[Any, ...]]
    image_indices: tuple[int, ...]

    @property
    def default_position(self) -> int:
        generated = np.asarray(self.session.generated_positions, dtype=np.int64)
        return int(generated[0]) if generated.size else self.selectable_positions[0]

    @property
    def default_layer(self) -> int:
        return self.layers[0]

    @property
    def default_image_index(self) -> int | None:
        return self.image_indices[0] if self.image_indices else None

    def heads_for(self, layer: int) -> tuple[Any, ...]:
        try:
            return self.heads_by_layer[int(layer)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Layer {layer!r} is not available in this session") from exc


@dataclass(frozen=True)
class WorkbenchRender:
    overlay: Any
    context_html: str
    mass_summary: dict[str, float]


def build_selector_state(session: Any) -> SelectorState:
    selectable_positions = tuple(int(position) for position in session.selectable_positions)
    layers = tuple(sorted(int(layer) for layer in session.layers))
    if not selectable_positions:
        raise ValueError("The session contains no text tokens to inspect")
    if not layers:
        raise ValueError("The session contains no full-attention layers")
    image_indices = tuple(range(len(session.images)))
    if len(image_indices) != len(session.visual_grid_hws):
        raise ValueError("The session image/grid counts do not match")

    heads_by_layer = {}
    for layer in layers:
        n_heads = int(session.layers[layer].visual.shape[0])
        heads_by_layer[layer] = ("Mean", *range(n_heads))
    return SelectorState(
        session,
        selectable_positions,
        layers,
        heads_by_layer,
        image_indices,
    )


def _gallery_images(value: Any) -> tuple[Any, ...]:
    if not value:
        return ()
    images = []
    for item in value:
        if isinstance(item, (tuple, list)):
            if not item:
                raise ValueError("An uploaded image entry is empty")
            item = item[0]
        images.append(item)
    return tuple(images)


def append_image_resize_specs(
    image_gallery: Any,
    existing_specs: Any = None,
) -> list[dict[str, Any]]:
    """Append resize state for new Gallery images without touching prior state."""
    images = _gallery_images(image_gallery)
    specs = [dict(spec) for spec in existing_specs or ()]
    if len(images) < len(specs):
        raise ValueError("Image deletion must be handled by its Gallery delete event")
    for image in images[len(specs) :]:
        if not isinstance(image, PILImage.Image):
            raise TypeError("Gallery entries must be PIL images")
        specs.append(
            {
                "id": str(uuid4()),
                "width": int(image.width),
                "height": int(image.height),
            }
        )
    return specs


def _resize_sizes(resize_specs: Any, image_count: int) -> tuple[tuple[int, int], ...]:
    specs = list(resize_specs or ())
    if len(specs) != int(image_count):
        raise ValueError(
            f"Resize settings must match the {image_count} input images; "
            f"actual {len(specs)}"
        )
    return tuple(
        (
            _positive_dimension(spec.get("width"), f"Image {index + 1} resize width"),
            _positive_dimension(spec.get("height"), f"Image {index + 1} resize height"),
        )
        for index, spec in enumerate(specs)
    )


def run_inference(
    cache: SessionCache,
    extractor: Callable[..., Any],
    model: Any,
    processor: Any,
    image_gallery: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    device: str,
    resize_specs: Any,
) -> tuple[str, SelectorState]:
    """Validate inputs, perform one inference, and cache the resulting session."""
    images = _gallery_images(image_gallery)
    has_text = bool(user_prompt and user_prompt.strip())
    if not images and not has_text:
        raise ValueError("At least one image or non-empty user prompt is required")
    resize_sizes = _resize_sizes(resize_specs, len(images))

    session = extractor(
        model=model,
        processor=processor,
        images=images,
        user_prompt=user_prompt,
        system_prompt=system_prompt or "",
        max_new_tokens=int(max_new_tokens),
        device=device,
        resize_sizes=resize_sizes,
    )
    selector = build_selector_state(session)
    return cache.put(session), selector


def _positive_dimension(value: Any, label: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric < 1:
        raise ValueError(f"{label} must be a positive integer")
    return round(numeric)


def _normalize_image_roots(
    image_roots: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    configured_roots = image_roots or (Path.cwd(),)
    roots = []
    for configured_root in configured_roots:
        try:
            root = Path(configured_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Image root does not exist: {configured_root}") from exc
        if not root.is_dir():
            raise ValueError(f"Image root is not a directory: {root}")
        roots.append(root)
    return tuple(roots)


def load_server_image(
    image_path: str | Path,
    image_roots: Sequence[str | Path] | None = None,
) -> PILImage.Image:
    """Load one server-side image constrained to configured filesystem roots."""
    raw_path = str(image_path).strip()
    if not raw_path:
        raise ValueError("A server image path is required")

    roots = _normalize_image_roots(image_roots)
    requested = Path(raw_path).expanduser()
    candidate = requested if requested.is_absolute() else roots[0] / requested
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Server image path does not exist: {candidate}") from exc
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Server image path is outside the allowed image roots")
    if not resolved.is_file():
        raise ValueError(f"Server image path is not a file: {resolved}")

    try:
        with PILImage.open(resolved) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as exc:
        raise ValueError(f"Server image path is not a valid image: {resolved}") from exc


def render_cached_session(
    cache: SessionCache,
    session_id: str,
    selected_position: int,
    layer: int,
    head: Any,
    image_index: int | None,
    opacity: float,
    *,
    renderer: Callable[..., Any],
) -> Any:
    """Render an existing session without invoking the model."""
    session = cache.get(session_id)
    return renderer(
        session,
        int(selected_position),
        int(layer),
        _coerce_head(head),
        None if image_index is None else int(image_index),
        float(opacity),
    )


def render_workbench_selection(
    session: Any,
    selected_position: int,
    layer: int,
    head: Any,
    image_index: int | None,
    opacity: float,
) -> WorkbenchRender:
    """Adapt a raw renderer slice to the workbench's causal presentation."""
    from .render import attention_masses, render_selection, split_visual_attention

    selection = render_selection(
        session,
        selected_position,
        layer,
        head,
        image_index,
        opacity,
    )
    positions = np.asarray(session.selectable_positions)
    matches = np.flatnonzero(positions == int(selected_position))
    if matches.size != 1:
        raise ValueError(f"Text token position {selected_position} is not available")
    causal_mask = np.asarray(session.causal_context_mask[int(matches[0])], dtype=bool)
    if causal_mask.shape != selection.context.shape:
        raise ValueError("Causal context mask does not match context attention")

    causal_context = np.where(causal_mask, selection.context, 0.0)
    masses = attention_masses(causal_context, session.context_key_positions, session.tokens)
    visual_parts = split_visual_attention(selection.visual, session.visual_grid_hws)
    image_masses = {
        f"image_{index + 1}": float(np.sum(values, dtype=np.float64))
        for index, values in enumerate(visual_parts)
    }
    if image_masses:
        masses = {
            **image_masses,
            "image_total": float(np.sum(selection.visual, dtype=np.float64)),
            **masses,
        }
    context_markup = _causal_context_html(
        session,
        causal_mask,
        selection.context,
        selection.vmax,
    )
    return WorkbenchRender(selection.overlay, context_markup, masses)


def _causal_context_html(
    session: Any,
    causal_mask: np.ndarray,
    values: np.ndarray,
    vmax: float,
) -> str:
    tokens = {int(token.absolute_position): token for token in session.tokens}
    parts = ['<div class="context-stream" aria-label="Prior text attention">']
    for index, position in enumerate(session.context_key_positions):
        if not causal_mask[index]:
            continue
        token = tokens[int(position)]
        intensity = 0.0 if vmax == 0 else float(np.clip(values[index] / vmax, 0, 1))
        red = round(255 * intensity)
        blue = round(255 * (1.0 - intensity))
        tooltip = escape(
            f"Raw piece: {token.raw_piece}\nDecoded: {token.decoded_preview}\n"
            f"Token ID: {token.token_id}\nPosition: {int(position)}\n"
            f"Raw attention: {float(values[index]):.8g}",
            quote=True,
        )
        parts.append(
            f'<span class="context-token context-{escape(token.segment)}" '
            f'data-position="{int(position)}" title="{tooltip}" '
            f'style="background:rgb({red},80,{blue});color:white">'
            f"{escape(_token_display_piece(token))}</span>"
        )
    parts.append("</div>")
    return "".join(parts)


def _coerce_head(head: Any) -> Any:
    if isinstance(head, str):
        if head.lower() == "mean":
            return "Mean"
        try:
            return int(head)
        except ValueError:
            pass
    return head


def _visible_piece(piece: str) -> str:
    if piece == "":
        return "[empty]"
    return piece.replace("\n", "\\n").replace("\t", "\\t")


def _token_display_piece(token: Any) -> str:
    decoded = str(getattr(token, "decoded_preview", ""))
    return _visible_piece(decoded) if decoded else "[byte]"


def token_ribbon_html(session: Any, selected_position: int | None = None) -> str:
    """Render all non-image tokens without changing their absolute positions."""
    parts = ['<div class="sequence-ribbon" aria-label="Full token sequence">']
    for token in session.tokens:
        if token.is_image:
            continue
        position = int(token.absolute_position)
        label = escape(_token_display_piece(token))
        tooltip = escape(
            "\n".join(
                (
                    f"Raw piece: {token.raw_piece}",
                    f"Decoded: {token.decoded_preview}",
                    f"Token ID: {token.token_id}",
                    f"Position: {position}",
                    f"Segment: {token.segment}",
                )
            ),
            quote=True,
        )
        classes = ["token", f"token-{escape(token.segment)}"]
        if token.is_special:
            classes.append("token-special")
        if position == selected_position:
            classes.append("token-selected")
        attrs = [
            f'class="{" ".join(classes)}"',
            f'data-position="{position}"',
            f'title="{tooltip}"',
        ]
        classes.append("token-selectable")
        attrs[0] = f'class="{" ".join(classes)}"'
        attrs.append(f'aria-label="Select text token at position {position}"')
        attrs.append(f'aria-pressed="{"true" if position == selected_position else "false"}"')
        if position == selected_position:
            attrs.append('aria-current="true"')
        parts.append(f"<button type=\"button\" {' '.join(attrs)}>{label}</button>")
    parts.append("</div>")
    return "".join(parts)


def build_app(
    model: Any,
    processor: Any,
    *,
    device: str = "cuda",
    cache: SessionCache | None = None,
    extractor: Callable[..., Any] | None = None,
    image_roots: Sequence[str | Path] | None = None,
):
    """Build the Gradio Blocks graph without running model inference."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required to build the workbench") from exc

    if extractor is None:
        from .attention import extract_attention

        extractor = extract_attention
    cache = cache if cache is not None else SessionCache()
    allowed_image_roots = _normalize_image_roots(image_roots)

    def render_outputs(session_id, selected_position, layer, head, image_index, opacity):
        selection = render_cached_session(
            cache,
            session_id,
            selected_position,
            layer,
            head,
            image_index,
            opacity,
            renderer=render_workbench_selection,
        )
        session = cache.get(session_id)
        if image_index is None:
            input_status = "Text-only input"
        else:
            selected_image = int(image_index)
            source_width, source_height = session.images[selected_image].size
            model_width, model_height = session.model_input_sizes[selected_image]
            input_status = (
                f"Image {selected_image + 1}/{len(session.images)} | "
                f"Input {source_width} x {source_height} -> "
                f"{model_width} x {model_height}"
            )
        return (
            token_ribbon_html(session, int(selected_position)),
            selection.overlay,
            selection.context_html,
            json.dumps(selection.mass_summary, ensure_ascii=False, indent=2),
            (
                f"Position {selected_position} | Layer {layer} | Head {head} | {input_status}"
            ),
        )

    def on_run(
        image_gallery,
        system_prompt,
        user_prompt,
        resize_specs,
        max_new_tokens,
    ):
        session_id, selector = run_inference(
            cache,
            extractor,
            model,
            processor,
            image_gallery,
            system_prompt,
            user_prompt,
            max_new_tokens,
            device,
            resize_specs=resize_specs,
        )
        position = selector.default_position
        layer = selector.default_layer
        head = "Mean"
        image_index = selector.default_image_index
        ribbon, spatial, context, mass, status = render_outputs(
            session_id, position, layer, head, image_index, 0.55
        )
        image_choices = [
            (f"Image {index + 1}", index) for index in selector.image_indices
        ]
        return (
            session_id,
            ribbon,
            position,
            gr.update(choices=list(selector.layers), value=layer),
            gr.update(choices=list(selector.heads_for(layer)), value=head),
            gr.update(choices=image_choices, value=image_index),
            spatial,
            context,
            mass,
            status,
        )

    def on_layer(session_id, selected_position, layer, image_index, opacity):
        selector = build_selector_state(cache.get(session_id))
        heads = selector.heads_for(int(layer))
        outputs = render_outputs(
            session_id,
            selected_position,
            layer,
            "Mean",
            image_index,
            opacity,
        )
        return gr.update(choices=list(heads), value="Mean"), *outputs

    def on_token_click(click_value, session_id, layer, head, image_index, opacity):
        position = int(str(click_value).split(":", 1)[0])
        return position, *render_outputs(
            session_id,
            position,
            layer,
            head,
            image_index,
            opacity,
        )

    def resize_updates(resize_specs, selected_id):
        for index, spec in enumerate(resize_specs or ()):
            if spec.get("id") == selected_id:
                image_number = index + 1
                return (
                    gr.update(
                        value=spec.get("width"),
                        label=f"Resize width · Image {image_number}",
                    ),
                    gr.update(
                        value=spec.get("height"),
                        label=f"Resize height · Image {image_number}",
                    ),
                )
        return (
            gr.update(value=None, label="Resize width"),
            gr.update(value=None, label="Resize height"),
        )

    def on_image_upload(image_gallery, resize_specs):
        specs = append_image_resize_specs(image_gallery, resize_specs)
        selected_id = specs[-1]["id"] if specs else None
        return specs, selected_id, *resize_updates(specs, selected_id)

    def on_image_change(image_gallery):
        if _gallery_images(image_gallery):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return [], None, *resize_updates([], None)

    def on_image_delete(resize_specs, selected_id, event):
        specs = [dict(spec) for spec in resize_specs or ()]
        index = int(event._data["index"])
        if not 0 <= index < len(specs):
            raise ValueError(f"Deleted image index {index} is unavailable")
        deleted_id = specs[index]["id"]
        specs.pop(index)
        if selected_id == deleted_id:
            selected_id = specs[min(index, len(specs) - 1)]["id"] if specs else None
        return specs, selected_id, *resize_updates(specs, selected_id)

    on_image_delete.__annotations__["event"] = gr.EventData

    def on_image_select(resize_specs, event):
        index = int(event.index)
        specs = list(resize_specs or ())
        if not 0 <= index < len(specs):
            raise ValueError(f"Selected image index {index} is unavailable")
        selected_id = specs[index]["id"]
        return selected_id, *resize_updates(specs, selected_id)

    on_image_select.__annotations__["event"] = gr.SelectData

    def update_resize_value(value, resize_specs, selected_id, field):
        specs = [dict(spec) for spec in resize_specs or ()]
        for spec in specs:
            if spec.get("id") == selected_id:
                spec[field] = value
                return specs
        raise ValueError("Select an input image before changing its resize settings")

    def on_resize_width(value, resize_specs, selected_id):
        return update_resize_value(value, resize_specs, selected_id, "width")

    def on_resize_height(value, resize_specs, selected_id):
        return update_resize_value(value, resize_specs, selected_id, "height")

    def on_server_image_load(image_path, image_gallery, resize_specs):
        image = load_server_image(image_path, allowed_image_roots)
        width, height = image.size
        images = [*_gallery_images(image_gallery), image]
        specs = append_image_resize_specs(images, resize_specs)
        selected_id = specs[-1]["id"]
        return (
            gr.update(value=images, selected_index=len(images) - 1),
            specs,
            selected_id,
            *resize_updates(specs, selected_id),
            f"Server image loaded ({width} x {height})",
        )

    css = """
    .gradio-container { width:90vw !important; max-width:none !important; margin-inline:auto !important; box-sizing:border-box !important; }
    .sequence-ribbon { display:flex; flex-wrap:wrap; gap:4px; padding:10px; border:1px solid #d1d5db; background:#fff; }
    .token { font-family:ui-monospace,monospace; font-size:12px; line-height:1.35; padding:4px 6px; border:1px solid #d1d5db; border-radius:3px; color:#111827 !important; background:#f8fafc !important; white-space:pre-wrap; }
    .context-token { white-space:pre-wrap; }
    button.token { appearance:none; width:auto; min-width:0; margin:0; box-shadow:none; }
    .token-selectable { cursor:pointer; }
    .token-generated { border-color:#0f766e; color:#064e3b !important; background:#ecfdf5 !important; }
    .token-selectable:hover { background:#dbeafe !important; }
    .token-selectable:focus-visible { outline:2px solid #0f766e; outline-offset:2px; }
    .token-special { background:#fef3c7 !important; }
    .token-selected, .token-selected:hover, .token-selected:focus-visible { color:#fff !important; background:#115e59 !important; border-color:#115e59; }
    #primary-input-row { height:32vh !important; align-items:stretch !important; }
    #image-input-column, #prompt-input-column { flex:1 1 0 !important; width:50% !important; min-width:0 !important; height:100% !important; overflow:hidden !important; }
    #image-input-column { display:grid !important; grid-template-rows:minmax(0,1fr) min-content !important; gap:0 !important; }
    #attention-image-input { width:100% !important; height:100% !important; min-height:0 !important; overflow:hidden !important; }
    #attention-image-input .image-container { min-height:0 !important; flex:1 1 auto !important; }
    #attention-image-input img { object-fit:contain !important; }
    #server-image-path-row { width:100% !important; min-height:0 !important; padding-top:3px !important; gap:3px !important; flex-wrap:nowrap !important; align-items:stretch !important; }
    #server-image-path-input { flex:1 1 auto !important; min-width:0 !important; }
    #load-server-image-button { flex:0 0 auto !important; min-width:5rem !important; margin:0 !important; }
    #prompt-input-column { display:grid !important; grid-template-rows:min-content minmax(0,1fr) !important; gap:0 !important; }
    #system-prompt-row, #user-prompt-row { width:100% !important; min-height:0 !important; gap:0 !important; flex-wrap:nowrap !important; }
    #user-prompt-row { height:100% !important; overflow:hidden !important; }
    #system-prompt-row > .form, #user-prompt-row > .form { width:100% !important; min-height:0 !important; border:0 !important; box-shadow:none !important; background:transparent !important; box-sizing:border-box !important; }
    #user-prompt-row > .form { height:100% !important; overflow:hidden !important; }
    #system-prompt-input { padding-bottom:3px !important; }
    #user-prompt-input { height:100% !important; min-height:0 !important; padding-top:3px !important; padding-bottom:10px !important; overflow:hidden !important; }
    #secondary-input-row { height:auto !important; min-height:0 !important; align-items:stretch !important; }
    #resize-input-column, #execution-input-column { flex:1 1 0 !important; width:50% !important; min-width:0 !important; height:auto !important; align-self:stretch !important; }
    #resize-input-row { width:100% !important; height:100% !important; align-items:stretch !important; }
    #resize-width-input, #resize-height-input { height:100% !important; }
    #execution-input-column { display:grid !important; grid-template-columns:minmax(0,1fr) !important; grid-template-rows:auto auto !important; align-items:stretch !important; gap:0 !important; }
    #execution-input-column > .form { width:100% !important; height:100% !important; min-height:0 !important; box-sizing:border-box !important; }
    #max-new-tokens { width:100% !important; height:100% !important; min-height:0 !important; max-height:none !important; }
    #run-button { width:100% !important; height:100% !important; min-height:0 !important; max-height:none !important; flex:none !important; margin:0 !important; }
    #system-prompt-input input { line-height:1.5 !important; overflow-y:auto !important; }
    #user-prompt-input > label { display:flex !important; flex-direction:column !important; height:100% !important; min-height:0 !important; }
    #user-prompt-input .input-container { flex:1 1 auto !important; width:100% !important; height:100% !important; min-height:0 !important; align-items:stretch !important; }
    #user-prompt-input textarea { flex:1 1 auto !important; height:100% !important; min-height:0 !important; max-height:none !important; overflow-y:auto !important; resize:none !important; }
    @media (max-width:48em) {
      .gradio-container { width:96vw !important; }
      #primary-input-row, #secondary-input-row { flex-direction:column !important; flex-wrap:nowrap !important; }
      #image-input-column, #prompt-input-column { width:100% !important; height:32vh !important; }
      #resize-input-column, #execution-input-column { flex:0 0 auto !important; width:100% !important; height:auto !important; }
    }
    #token-click-bridge { display:none !important; }
    """
    with gr.Blocks(title="VLM Attention Workbench", css=css, js=TOKEN_RIBBON_JS) as demo:
        session_state = gr.State("")
        query_state = gr.State(None)
        resize_specs_state = gr.State([])
        selected_input_image_state = gr.State(None)
        token_click = gr.Textbox(
            value="",
            show_label=False,
            container=False,
            interactive=True,
            elem_id="token-click-bridge",
        )
        gr.Markdown("# VLM Attention Workbench")
        with gr.Row(equal_height=True, elem_id="primary-input-row"):
            with gr.Column(scale=1, elem_id="image-input-column"):
                image_input = gr.Gallery(
                    type="pil",
                    label="Images",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    interactive=True,
                    elem_id="attention-image-input",
                )
                with gr.Row(elem_id="server-image-path-row"):
                    server_image_path = gr.Textbox(
                        label="Server image path",
                        placeholder="Server image path",
                        lines=1,
                        max_lines=1,
                        show_label=False,
                        container=False,
                        elem_id="server-image-path-input",
                    )
                    load_server_image_button = gr.Button(
                        "Load",
                        elem_id="load-server-image-button",
                    )
            with gr.Column(scale=1, elem_id="prompt-input-column"):
                with gr.Row(elem_id="system-prompt-row"):
                    system_input = gr.Textbox(
                        label="System prompt",
                        lines=1,
                        max_lines=1,
                        elem_id="system-prompt-input",
                    )
                with gr.Row(elem_id="user-prompt-row"):
                    user_input = gr.Textbox(
                        label="User prompt",
                        lines=4,
                        max_lines=4,
                        elem_id="user-prompt-input",
                    )
        with gr.Row(equal_height=True, elem_id="secondary-input-row"):
            with gr.Column(
                scale=1,
                elem_id="resize-input-column",
            ), gr.Row(equal_height=True, elem_id="resize-input-row"):
                resize_width = gr.Number(
                    label="Resize width",
                    precision=0,
                    minimum=1,
                    elem_id="resize-width-input",
                )
                resize_height = gr.Number(
                    label="Resize height",
                    precision=0,
                    minimum=1,
                    elem_id="resize-height-input",
                )
            with gr.Column(scale=1, elem_id="execution-input-column"):
                max_tokens = gr.Slider(
                    1,
                    1024,
                    value=128,
                    step=1,
                    label="Max new tokens",
                    elem_id="max-new-tokens",
                )
                run_button = gr.Button(
                    "Run",
                    variant="primary",
                    elem_id="run-button",
                )
        status = gr.Markdown("Ready")
        ribbon = gr.HTML(
            '<div class="sequence-ribbon"></div>',
            elem_id="token-ribbon",
        )
        with gr.Row():
            layer = gr.Dropdown(label="Full-attention layer", choices=[])
            head = gr.Dropdown(label="Head", choices=[])
            image_index = gr.Dropdown(label="Overlay image", choices=[])
            opacity = gr.Slider(0, 1, value=0.55, label="Overlay opacity")
        with gr.Tabs():
            with gr.Tab("Spatial"):
                spatial = gr.Image(label="Attention overlay", interactive=False)
            with gr.Tab("Context"):
                context = gr.HTML()
        mass = gr.Code(
            label="Raw attention mass",
            language="json",
            interactive=False,
            lines=6,
            show_line_numbers=False,
        )

        run_button.click(
            on_run,
            inputs=[
                image_input,
                system_input,
                user_input,
                resize_specs_state,
                max_tokens,
            ],
            outputs=[
                session_state,
                ribbon,
                query_state,
                layer,
                head,
                image_index,
                spatial,
                context,
                mass,
                status,
            ],
            concurrency_limit=1,
        )
        image_input.change(
            on_image_change,
            image_input,
            [
                resize_specs_state,
                selected_input_image_state,
                resize_width,
                resize_height,
            ],
            queue=False,
        )
        image_input.upload(
            on_image_upload,
            [image_input, resize_specs_state],
            [
                resize_specs_state,
                selected_input_image_state,
                resize_width,
                resize_height,
            ],
            queue=False,
        )
        image_input.delete(
            on_image_delete,
            [resize_specs_state, selected_input_image_state],
            [
                resize_specs_state,
                selected_input_image_state,
                resize_width,
                resize_height,
            ],
            queue=False,
        )
        image_input.select(
            on_image_select,
            resize_specs_state,
            [selected_input_image_state, resize_width, resize_height],
            queue=False,
        )
        resize_width.input(
            on_resize_width,
            [resize_width, resize_specs_state, selected_input_image_state],
            resize_specs_state,
            queue=False,
        )
        resize_height.input(
            on_resize_height,
            [resize_height, resize_specs_state, selected_input_image_state],
            resize_specs_state,
            queue=False,
        )
        server_image_outputs = [
            image_input,
            resize_specs_state,
            selected_input_image_state,
            resize_width,
            resize_height,
            status,
        ]
        load_server_image_button.click(
            on_server_image_load,
            [server_image_path, image_input, resize_specs_state],
            server_image_outputs,
            queue=False,
        )
        server_image_path.submit(
            on_server_image_load,
            [server_image_path, image_input, resize_specs_state],
            server_image_outputs,
            queue=False,
        )
        common_inputs = [session_state, query_state, layer, head, image_index, opacity]
        common_outputs = [ribbon, spatial, context, mass, status]
        head.change(render_outputs, common_inputs, common_outputs, queue=False)
        image_index.change(render_outputs, common_inputs, common_outputs, queue=False)
        opacity.change(render_outputs, common_inputs, common_outputs, queue=False)
        layer.change(
            on_layer,
            [session_state, query_state, layer, image_index, opacity],
            [head, ribbon, spatial, context, mass, status],
            queue=False,
        )
        token_click.input(
            on_token_click,
            [token_click, session_state, layer, head, image_index, opacity],
            [query_state, ribbon, spatial, context, mass, status],
            queue=False,
        )
    return demo
