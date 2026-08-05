"""Command-line entry point for the Gradio attention workbench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def load_model(model_path: str, device: str):
    from .attention import load_model as load

    return load(model_path, device=device)


def build_app(model, processor, device: str, image_roots=None):
    from .app import build_app as build

    return build(model, processor, device=device, image_roots=image_roots)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vlm_attention_viz",
        description="Launch the VLM Attention Workbench.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--device", default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--image-root",
        action="append",
        help=(
            "Directory allowed for server-side image loading; repeat to allow "
            "multiple directories (default: current working directory)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    model, processor = load_model(args.model, device=args.device)
    demo = build_app(
        model,
        processor,
        device=args.device,
        image_roots=args.image_root,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
