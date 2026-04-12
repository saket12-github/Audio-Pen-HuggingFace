"""Transcription via faster-whisper (model load + decode)."""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import gradio as gr

from audio_utils import validate_audio
from config import (
    WHISPER_BEST_OF,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_INIT_NUM_WORKERS,
    WHISPER_MODEL_SIZE,
    WHISPER_VAD_FILTER,
    WHISPER_WITHOUT_TIMESTAMPS,
)
from helpers import report_progress, truncate_message

logger = logging.getLogger(__name__)

_whisper_model = None


def get_whisper_model():
    """Lazy-load Whisper once (saves RAM on Spaces until first use)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s beam=%s best_of=%s "
            "cpu_threads=%s init_workers=%s vad=%s without_ts=%s",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
            WHISPER_BEAM_SIZE,
            WHISPER_BEST_OF,
            WHISPER_CPU_THREADS or "auto",
            WHISPER_INIT_NUM_WORKERS,
            WHISPER_VAD_FILTER,
            WHISPER_WITHOUT_TIMESTAMPS,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=WHISPER_INIT_NUM_WORKERS,
        )
        logger.info("Whisper model loaded")
    return _whisper_model


def _approx_word_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return text.count(" ") + text.count("\n") + 1


def transcribe_audio(
    audio_path: Optional[str],
    progress: gr.Progress | None = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio to plain text. Returns (transcript, error_message).
    On success error_message is None.
    """
    t_wall0 = time.perf_counter()
    try:
        report_progress(progress, 0.02, "Validating audio…", logger)
        ok, msg = validate_audio(audio_path)
        if not ok:
            logger.warning("Validation failed: %s", msg)
            return None, msg

        assert audio_path is not None
        report_progress(progress, 0.08, "Loading transcription model…", logger)
        model = get_whisper_model()

        report_progress(progress, 0.12, "Transcribing (this may take a while)…", logger)
        logger.info("Starting transcribe path=%s", audio_path)

        t_decode = time.perf_counter()
        segments, info = model.transcribe(
            audio_path,
            language=None,
            task="transcribe",
            vad_filter=WHISPER_VAD_FILTER,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            without_timestamps=WHISPER_WITHOUT_TIMESTAMPS,
        )
        if info and getattr(info, "language", None):
            logger.info("Detected language: %s", info.language)

        parts: list[str] = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)

        transcript = " ".join(parts).strip()
        decode_s = time.perf_counter() - t_decode
        if not transcript:
            logger.warning("Empty transcript (no speech detected)")
            return None, "Warning: No speech detected. Try clearer audio or a different file."

        report_progress(progress, 0.48, "Transcription complete", logger)
        wall_s = time.perf_counter() - t_wall0
        logger.info(
            "Transcription decode %.2fs | total wall %.2fs | transcript: %s chars, ~%s words",
            decode_s,
            wall_s,
            len(transcript),
            _approx_word_count(transcript),
        )
        return transcript, None

    except OSError as e:
        err = truncate_message(f"Transcription failed (I/O): {e}")
        logger.exception("Transcribe I/O error")
        return None, err
    except RuntimeError as e:
        err = truncate_message(f"Transcription failed (runtime): {e}")
        logger.exception("Transcribe runtime error")
        return None, err
    except Exception as e:
        err = truncate_message(f"Transcription failed: {e}")
        logger.exception("Transcribe unexpected error")
        return None, err
