"""Audio file validation (used before transcription)."""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from config import MAX_AUDIO_MB, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)


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

    logger.info("Audio OK: ext=%s size_mb=%.2f", ext, size_mb)
    return True, "ok"
