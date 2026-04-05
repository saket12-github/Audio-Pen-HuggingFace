"""Audio validation and transcription via faster-whisper."""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import gradio as gr

from config import (
    MAX_AUDIO_MB,
    SUPPORTED_FORMATS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)

_whisper_model = None


def get_whisper_model():
    """Lazy-load Whisper once (saves RAM on Spaces until first use)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded")
    return _whisper_model


def validate_audio(audio_path: Optional[str]) -> Tuple[bool, str]:
    if audio_path is None:
        return False, "Error: No audio provided. Upload or record audio first."

    if not os.path.exists(audio_path):
        return False, "Error: Audio file not found on disk."

    try:
        raw_size = os.path.getsize(audio_path)
    except OSError as e:
        logger.exception("Could not stat audio file")
        return False, f"Error: Cannot read audio file ({e})"

    if raw_size == 0:
        return False, "Error: Audio file is empty."

    size_mb = raw_size / (1024 * 1024)
    if size_mb > MAX_AUDIO_MB:
        return (
            False,
            f"Error: File too large ({size_mb:.1f} MB). Maximum is {MAX_AUDIO_MB} MB.",
        )

    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        return (
            False,
            f"Error: Unsupported format '.{ext}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    logger.info("Audio OK: ext=%s size_mb=%.2f path=%s", ext, size_mb, audio_path)
    return True, "ok"


def transcribe_audio(
    audio_path: Optional[str],
    progress: gr.Progress | None = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio to plain text. Returns (transcript, error_message).
    On success error_message is None.
    """
    def p(fraction: float, desc: str) -> None:
        if progress is not None:
            progress(fraction, desc=desc)
        logger.info("%s", desc)

    try:
        p(0.02, "Validating audio…")
        ok, msg = validate_audio(audio_path)
        if not ok:
            logger.warning("Validation failed: %s", msg)
            return None, msg

        assert audio_path is not None
        p(0.08, "Loading transcription model…")
        model = get_whisper_model()

        p(0.12, "Transcribing (this may take a while)…")
        logger.info("Starting transcribe: %s", audio_path)

        segments, info = model.transcribe(
            audio_path,
            language=None,
            task="transcribe",
            vad_filter=True,
            beam_size=5,
        )
        if info and getattr(info, "language", None):
            logger.info("Detected language: %s", info.language)

        parts: list[str] = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)

        transcript = " ".join(parts).strip()
        if not transcript:
            logger.warning("Empty transcript (no speech detected)")
            return None, "Warning: No speech detected. Try clearer audio or a different file."

        p(0.48, "Transcription complete")
        logger.info("Transcript length: %s chars, ~%s words", len(transcript), len(transcript.split()))
        return transcript, None

    except RuntimeError as e:
        err = f"Transcription failed (runtime): {str(e)[:200]}"
        logger.exception("Transcribe runtime error")
        return None, err
    except Exception as e:
        err = f"Transcription failed: {str(e)[:200]}"
        logger.exception("Transcribe unexpected error")
        return None, err
