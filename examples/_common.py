"""Shared helper for the examples: score with an LLM if a provider is given, otherwise prepare evidence only."""
from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--provider", default=None,
                        help="LLM provider (deepseek, gpt, claude, gemini, glm, qwen). "
                             "Omit to only retrieve and distill evidence (no API key needed).")
    parser.add_argument("--model", default=None, help="Model ID; recommended to set explicitly for reproducible runs.")
    parser.add_argument("--out", default=None, help="Output directory.")
    return parser.parse_args()


def llm_settings(args: argparse.Namespace) -> dict:
    if args.provider is None:
        print("No --provider given: running retrieval + evidence distillation only (prepare_only=True).")
        return {"prepare_only": True}
    return {"provider": args.provider, "model": args.model}
