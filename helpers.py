"""Small shared helpers (progress + user-facing messages)."""
from __future__ import annotations

import logging

import gradio as gr


def report_progress(
    progress: gr.Progress | None,
    fraction: float,
    desc: str,
    log: logging.Logger,
) -> None:
    if progress is not None:
        progress(fraction, desc=desc)
    log.info("%s", desc)


def truncate_message(msg: str, max_len: int = 500) -> str:
    msg = (msg or "").strip()
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 1].rstrip() + "…"
