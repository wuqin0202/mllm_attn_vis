# VLM Attention Workbench

Inspect how a Qwen-family vision-language model attends to image patches and prior text at any text-token position. The Gradio workbench keeps one inference result in memory and lets you move between input or generated text tokens, full-attention layers, and heads without rerunning the model.

## Scope

The first release supports:

- zero or one image;
- an optional system prompt and an optional user prompt, with at least one image or non-empty user prompt required;
- one assistant generation;
- Qwen2-VL, Qwen2.5-VL, and dense or MoE Qwen3.5-VL model layouts supported by the installed Transformers version;
- every tokenizer-produced token except the model's actual image-pad token;
- all captured full-attention layers and every returned attention head, plus an on-demand mean.

Multiple images, videos, multi-turn conversations, artifact persistence, and matrix comparison views are intentionally outside this release.

## Install

Python 3.10 or newer is required. Install the runtime dependencies in an environment suitable for the selected Torch/CUDA build:

```bash
pip install -r requirements.txt
```

## Launch

Use a Hugging Face model ID:

```bash
python -m vlm_attention_viz \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --image-root /path/to/server/images \
  --host 127.0.0.1 \
  --port 7860
```

Or point `--model` at a local Qwen checkpoint. Repeat `--image-root` to allow
multiple server directories. Relative server image paths are resolved from the
first configured root; absolute paths must remain inside one configured root.
Without this option, only the current working directory is allowed. View all
command options with:

```bash
python -m vlm_attention_viz --help
```

## Workflow

1. Optionally upload one image, or enter a permitted server-side path and select **Load**.
2. Optionally enter a system prompt and a user prompt. At least the image or user prompt must be present.
3. When using an image, enter the model input width and height. Uploading an image initializes both fields from its original dimensions; changing them independently may change the image's aspect ratio.
4. Choose the maximum generation length and select **Run**.
5. Select any visible input or generated text token by absolute sequence position.
6. Click a text token in the sequence ribbon, then select a full-attention layer and `Mean` or an individual head.
7. Inspect the Spatial view, Context view, opacity control, and raw attention mass summary.

For image input, the status shows the uploaded dimensions and the effective model-input dimensions. Each requested dimension is independently aligned to the nearest multiple of the model's `patch_size * spatial_merge_size`; this does not preserve the aspect ratio. The processor does not resize the aligned image a second time. For text-only input, the Spatial view is empty.

The sequence ribbon preserves one selectable cell per non-image tokenizer token, including input tokens, generated tokens, chat-template markers, role tokens, vision boundaries, EOS, whitespace, byte fallback pieces, and other special tokens. Labels use contextual streaming decode so byte-level markers are readable; raw tokenizer pieces remain available in tooltips. Repeated token pieces remain distinct because selection uses absolute position rather than display text.

## Attention Semantics

For a selected text token at absolute position `p`, the workbench reads attention query row `p`. The Context view applies the causal mask and therefore displays only earlier text positions. Input and generated tokens use the same absolute-position mapping.

Image-pad tokens stay in the model sequence and attention column mapping but are hidden from the token ribbon. All other tokens remain visible. For hybrid Qwen3.5 models, linear-attention layers still participate in inference but are not hooked, stored, or offered in the layer selector.

Spatial overlays use the selected image-patch slice's own min-max range, map low values to blue and high values to red, and blend the full map at the selected opacity. Context tokens keep their attention scale. Compare modalities, layers, or heads using raw values and modality masses rather than display color intensity.

Attention is an internal model weight, not a causal explanation or proof that a region caused the output. Head averaging can also hide distinct head behavior.

## Project Layout

```text
vlm_attention_viz/
  __init__.py       Public package surface
  __main__.py       Command-line entry
  attention.py      Model loading, token metadata, and attention capture
  render.py         Pure NumPy/PIL slice rendering
  app.py            Gradio workbench and bounded session cache
tests/
  test_attention.py
  test_render.py
  test_app.py
  test_cli.py
```

## Verification

Run the local gates with:

```bash
python -m unittest discover -s tests -v
python -m compileall vlm_attention_viz tests
python -m vlm_attention_viz --help
```

A real Qwen3.5 checkpoint and compatible GPU are required to validate live model output signatures and CUDA memory behavior. When available:

```bash
QWEN35_MODEL_PATH=/path/to/Qwen3.5-VL \
python -m unittest tests.test_attention.Qwen35SmokeTest -v
```

## License

MIT
