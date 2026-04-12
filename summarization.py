"""Transcript summarization via Together AI (chunk → reduce → dual outputs)."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import gradio as gr
import httpx

from config import (
    CONTEXT_TAIL_CHARS,
    SUMMARY_CHUNK_CHARS,
    SUMMARY_REDUCE_CHARS,
    SUMMARY_ROLLING_MAX_CHARS,
    TOGETHER_API_KEY_ENV,
    TOGETHER_API_URL,
    TOGETHER_CONNECT_TIMEOUT_S,
    TOGETHER_MAX_RETRIES,
    TOGETHER_MODEL,
    TOGETHER_TIMEOUT_S,
)
from helpers import report_progress, truncate_message

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    key = os.environ.get(TOGETHER_API_KEY_ENV, "").strip()
    return key or None


def _extract_message_text(data: dict[str, Any]) -> Optional[str]:
    try:
        choices = data.get("choices")
        if not choices:
            return None
        choice0 = choices[0] if isinstance(choices, list) else None
        if not isinstance(choice0, dict):
            return None
        msg = choice0.get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if content is None:
            return None
        return str(content).strip()
    except (TypeError, AttributeError):
        return None


def _together_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> Tuple[Optional[str], Optional[str]]:
    api_key = _get_api_key()
    if not api_key:
        return None, (
            f"Missing {TOGETHER_API_KEY_ENV}. Add it under Space Settings → Repository secrets."
        )

    payload = {
        "model": TOGETHER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(
        TOGETHER_TIMEOUT_S,
        connect=TOGETHER_CONNECT_TIMEOUT_S,
    )
    last_err: Optional[str] = None

    for attempt in range(TOGETHER_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(TOGETHER_API_URL, headers=headers, json=payload)
            if r.status_code == 401:
                return None, "Together API rejected the key (401). Check TOGETHER_API_KEY."
            if r.status_code == 429:
                last_err = "Together API rate limit (429). Retry shortly."
                time.sleep(2**attempt)
                continue
            if r.status_code >= 500:
                last_err = f"Together API server error ({r.status_code})."
                time.sleep(1 + attempt)
                continue
            if r.status_code == 400:
                try:
                    detail = r.json()
                except json.JSONDecodeError:
                    detail = r.text[:300]
                logger.warning("Together 400 response: %s", detail)
                return None, truncate_message(
                    f"Together API rejected the request (400). Check model name or payload. {detail!s}"
                )
            r.raise_for_status()
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                last_err = f"Together returned invalid JSON: {e}"
                logger.warning("%s", last_err)
            else:
                text = _extract_message_text(data)
                if text:
                    return text, None
                last_err = "Together API returned an empty or unrecognized response."
                logger.warning("%s raw_keys=%s", last_err, list(data.keys()) if isinstance(data, dict) else type(data))
        except httpx.TimeoutException as e:
            last_err = truncate_message(f"Together request timed out: {e}")
            logger.warning("%s", last_err)
        except httpx.HTTPStatusError as e:
            last_err = truncate_message(f"Together HTTP error: {e.response.status_code}")
            logger.exception("Together HTTPStatusError")
        except httpx.RequestError as e:
            last_err = truncate_message(f"Together request failed: {e}")
            logger.exception("Together RequestError")
        except (KeyError, IndexError, TypeError) as e:
            last_err = truncate_message(f"Unexpected Together response shape: {e}")
            logger.exception("Together parse error")

        if attempt < TOGETHER_MAX_RETRIES:
            time.sleep(1 + attempt)

    return None, last_err or "Together API call failed."


def chunk_transcript(text: str, max_chars: int) -> List[str]:
    """Split transcript into large chunks, preferring paragraph boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [text]

    chunks: List[str] = []
    buf = ""
    for p in paras:
        sep = "\n\n" if buf else ""
        candidate = f"{buf}{sep}{p}" if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(p) <= max_chars:
            buf = p
            continue
        start = 0
        while start < len(p):
            end = min(start + max_chars, len(p))
            piece = p[start:end].strip()
            if piece:
                chunks.append(piece)
            start = end
    if buf:
        chunks.append(buf)

    logger.info("Transcript split into %s chunk(s), max_chars=%s", len(chunks), max_chars)
    return chunks


def _tail_context(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[-n:].strip()


def _cap_rolling_notes(rolling: str, max_chars: int) -> str:
    rolling = rolling.strip()
    if len(rolling) <= max_chars:
        return rolling
    return rolling[-max_chars:].strip()


def summarize_chunk_with_context(
    chunk: str,
    chunk_index: int,
    num_chunks: int,
    prior_notes: str,
) -> Tuple[Optional[str], Optional[str]]:
    prior = prior_notes.strip()
    sys = (
        "You extract accurate, faithful notes from transcript excerpts. "
        "Use clear sentences and bullet points where helpful. "
        "Do not invent facts. If audio quality is unclear, say what is uncertain."
    )
    user = (
        f"This is part {chunk_index + 1} of {num_chunks} of a longer transcript.\n\n"
    )
    if prior:
        user += (
            "Context from earlier parts (may be truncated):\n"
            f"{prior}\n\n"
        )
    user += "Transcript excerpt:\n---\n" + chunk + "\n---\n\n"
    user += (
        "Summarize ONLY this excerpt. Include names, numbers, dates, and conclusions when present. "
        "Write so a later step can merge all parts into one coherent document."
    )
    return _together_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=2048,
        temperature=0.15,
    )


def reduce_notes(notes: List[str], max_chars: int) -> Tuple[str, Optional[str]]:
    """Merge many chunk notes into a single narrative; recurse if needed."""
    combined = "\n\n".join(n.strip() for n in notes if n.strip()).strip()
    if not combined:
        return "", "No intermediate notes to reduce."

    if len(combined) <= max_chars:
        sys = "You merge partial transcript summaries into one faithful synthesis."
        user = (
            "Below are section summaries from one continuous transcript (in order). "
            "Merge them into a single coherent narrative. "
            "Preserve important facts, ordering, and nuance. Do not add new claims.\n\n"
            f"{combined}"
        )
        out, err = _together_chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=4096,
            temperature=0.15,
        )
        if err:
            return "", err
        return (out or "").strip(), None

    logger.info("Reduce pass: combined notes too long (%s chars), splitting", len(combined))
    subchunks = chunk_transcript(combined, max_chars=max_chars)
    sub_notes: List[str] = []
    for i, sc in enumerate(subchunks):
        sys = "You compress partial summaries without losing key information."
        user = (
            f"Sub-block {i + 1} of {len(subchunks)} from a merged-summary document.\n"
            "Condense while keeping critical facts and structure hints.\n\n"
            f"{sc}"
        )
        part, err = _together_chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=2048,
            temperature=0.1,
        )
        if err:
            return "", err
        if part:
            sub_notes.append(part)
    return reduce_notes(sub_notes, max_chars=max_chars)


def generate_concise(synthesis: str) -> Tuple[Optional[str], Optional[str]]:
    sys = "You write tight, accurate executive summaries."
    user = (
        "From the following synthesis of a full transcript, write a concise summary:\n"
        "- 2–5 short paragraphs OR up to 8 bullet points\n"
        "- Cover the full scope (not only the beginning)\n"
        "- No preamble, no 'this transcript discusses'\n\n"
        f"{synthesis.strip()}"
    )
    return _together_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=1024,
        temperature=0.2,
    )


def generate_detailed(synthesis: str) -> Tuple[Optional[str], Optional[str]]:
    sys = "You produce structured, thorough summaries grounded in the provided text."
    user = (
        "From the following synthesis of a full transcript, produce a detailed structured summary.\n"
        "Use this outline (use headings exactly):\n"
        "## Overview\n"
        "## Key points\n"
        "## Important details\n"
        "## Conclusions / action items\n\n"
        "Represent the entire content, not just the opening. Do not invent information.\n\n"
        f"{synthesis.strip()}"
    )
    return _together_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=4096,
        temperature=0.2,
    )


def summarize_transcript(
    transcript: str,
    progress: gr.Progress | None = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Full pipeline: chunk → per-chunk summaries with rolling context → reduce → concise + detailed.
    Returns (concise, detailed, error). If error is set, concise/detailed may be None.
    """
    t = (transcript or "").strip()
    if not t:
        return None, None, "Nothing to summarize (empty transcript)."

    report_progress(progress, 0.52, "Checking Together API configuration…", logger)
    if not _get_api_key():
        return None, None, (
            f"Missing {TOGETHER_API_KEY_ENV}. Summarization is disabled until the secret is set."
        )

    chunks = chunk_transcript(t, max_chars=SUMMARY_CHUNK_CHARS)
    if not chunks:
        return None, None, "Could not chunk transcript."

    rolling = ""
    partial_notes: List[str] = []
    n = len(chunks)
    base = 0.55
    span = 0.38

    for i, ch in enumerate(chunks):
        frac = base + span * (i / max(n, 1))
        report_progress(progress, frac, f"Summarizing transcript part {i + 1}/{n}…", logger)
        prior = _tail_context(rolling, CONTEXT_TAIL_CHARS)
        note, err = summarize_chunk_with_context(ch, i, n, prior)
        if err:
            return None, None, err
        if note:
            partial_notes.append(note)
            rolling = (rolling + "\n\n" + note).strip()
            rolling = _cap_rolling_notes(rolling, SUMMARY_ROLLING_MAX_CHARS)

    if n == 1 and partial_notes:
        synthesis = partial_notes[0].strip()
        report_progress(progress, 0.9, "Single segment: using chunk summary as synthesis…", logger)
    else:
        report_progress(progress, 0.9, "Merging section summaries…", logger)
        synthesis, err = reduce_notes(partial_notes, max_chars=SUMMARY_REDUCE_CHARS)
        if err:
            return None, None, err
        if not synthesis.strip():
            return None, None, "Summarization produced an empty synthesis."

    report_progress(progress, 0.93, "Generating concise summary…", logger)
    concise, err_c = generate_concise(synthesis)
    if err_c:
        return None, None, err_c

    report_progress(progress, 0.97, "Generating detailed summary…", logger)
    detailed, err_d = generate_detailed(synthesis)
    if err_d:
        return concise, None, err_d

    report_progress(progress, 1.0, "Summarization complete", logger)
    return concise, detailed, None
