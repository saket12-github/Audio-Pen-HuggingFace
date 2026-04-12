"""Transcript summarization via Together AI (chunk → reduce → dual outputs)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import gradio as gr
import httpx

from config import (
    CONTEXT_TAIL_CHARS,
    ENABLE_SUMMARY_CACHE,
    PIPELINE_SUMMARY_CACHE_MAX,
    SUMMARY_CHUNK_CHARS,
    SUMMARY_MAX_PARALLEL,
    SUMMARY_REDUCE_CHARS,
    TOGETHER_API_KEY_ENV,
    TOGETHER_API_URL,
    TOGETHER_CONNECT_TIMEOUT_S,
    TOGETHER_MAX_RETRIES,
    TOGETHER_MODEL,
    TOGETHER_TIMEOUT_S,
)
from helpers import report_progress, truncate_message

logger = logging.getLogger(__name__)

_summary_cache_lock = threading.Lock()
_summary_cache: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()


def _cache_get(key: str) -> Optional[Tuple[str, str]]:
    if not ENABLE_SUMMARY_CACHE:
        return None
    with _summary_cache_lock:
        hit = _summary_cache.pop(key, None)
        if hit is None:
            return None
        _summary_cache[key] = hit
        return hit


def _cache_put(key: str, concise: str, detailed: str) -> None:
    if not ENABLE_SUMMARY_CACHE:
        return
    with _summary_cache_lock:
        _summary_cache[key] = (concise, detailed)
        _summary_cache.move_to_end(key)
        while len(_summary_cache) > PIPELINE_SUMMARY_CACHE_MAX:
            _summary_cache.popitem(last=False)


def _transcript_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _together_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        TOGETHER_TIMEOUT_S,
        connect=TOGETHER_CONNECT_TIMEOUT_S,
    )


def _together_headers() -> dict[str, str]:
    api_key = _get_api_key()
    assert api_key
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _together_chat_async(
    client: httpx.AsyncClient,
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
    last_err: Optional[str] = None

    for attempt in range(TOGETHER_MAX_RETRIES + 1):
        try:
            r = await client.post(
                TOGETHER_API_URL,
                headers=_together_headers(),
                json=payload,
            )
            if r.status_code == 401:
                return None, "Together API rejected the key (401). Check TOGETHER_API_KEY."
            if r.status_code == 429:
                last_err = "Together API rate limit (429). Retry shortly."
                await asyncio.sleep(2**attempt)
                continue
            if r.status_code >= 500:
                last_err = f"Together API server error ({r.status_code})."
                await asyncio.sleep(1 + attempt)
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
                logger.warning(
                    "%s raw_keys=%s",
                    last_err,
                    list(data.keys()) if isinstance(data, dict) else type(data),
                )
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
            await asyncio.sleep(1 + attempt)

    return None, last_err or "Together API call failed."


def _together_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> Tuple[Optional[str], Optional[str]]:
    """Synchronous single Together call (used where no async client is available)."""

    async def _run() -> Tuple[Optional[str], Optional[str]]:
        async with httpx.AsyncClient(timeout=_together_timeout()) as client:
            return await _together_chat_async(client, messages, max_tokens, temperature)

    return asyncio.run(_run())


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


def _prior_transcript_tail(chunks: List[str], index: int) -> str:
    """Overlap from the previous raw transcript chunk (enables parallel map calls)."""
    if index <= 0:
        return ""
    return _tail_context(chunks[index - 1], CONTEXT_TAIL_CHARS)


async def _summarize_chunk_async(
    client: httpx.AsyncClient,
    chunk: str,
    chunk_index: int,
    num_chunks: int,
    prior: str,
) -> Tuple[Optional[str], Optional[str]]:
    prior = prior.strip()
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
            "Context from the end of the previous transcript segment (may be truncated):\n"
            f"{prior}\n\n"
        )
    user += "Transcript excerpt:\n---\n" + chunk + "\n---\n\n"
    user += (
        "Summarize ONLY this excerpt. Include names, numbers, dates, and conclusions when present. "
        "Write so a later step can merge all parts into one coherent document."
    )
    return await _together_chat_async(
        client,
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=2048,
        temperature=0.15,
    )


async def _reduce_notes_async(
    client: httpx.AsyncClient,
    notes: List[str],
    max_chars: int,
    sem: asyncio.Semaphore,
) -> Tuple[str, Optional[str]]:
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
        out, err = await _together_chat_async(
            client,
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=4096,
            temperature=0.15,
        )
        if err:
            return "", err
        return (out or "").strip(), None

    logger.info("Reduce pass: combined notes too long (%s chars), splitting", len(combined))
    subchunks = chunk_transcript(combined, max_chars=max_chars)

    async def _compress_one(i: int, sc: str) -> Tuple[int, Optional[str], Optional[str]]:
        async with sem:
            sys = "You compress partial summaries without losing key information."
            user = (
                f"Sub-block {i + 1} of {len(subchunks)} from a merged-summary document.\n"
                "Condense while keeping critical facts and structure hints.\n\n"
                f"{sc}"
            )
            part, err = await _together_chat_async(
                client,
                [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                max_tokens=2048,
                temperature=0.1,
            )
            return i, part, err

    sub_results = await asyncio.gather(
        *(_compress_one(i, sc) for i, sc in enumerate(subchunks)),
        return_exceptions=True,
    )
    sub_notes: List[str] = []
    ordered: List[Tuple[int, Optional[str], Optional[str]]] = []
    for res in sub_results:
        if not isinstance(res, tuple):
            return "", truncate_message(f"Parallel reduce failed: {res}")
        ordered.append(res)
    ordered.sort(key=lambda t: t[0])
    for _, part, err in ordered:
        if err:
            return "", err
        if part:
            sub_notes.append(part)

    return await _reduce_notes_async(client, sub_notes, max_chars=max_chars, sem=sem)


async def _generate_concise_async(
    client: httpx.AsyncClient,
    synthesis: str,
) -> Tuple[Optional[str], Optional[str]]:
    sys = "You write tight, accurate executive summaries."
    user = (
        "From the following synthesis of a full transcript, write a concise summary:\n"
        "- 2–5 short paragraphs OR up to 8 bullet points\n"
        "- Cover the full scope (not only the beginning)\n"
        "- No preamble, no 'this transcript discusses'\n\n"
        f"{synthesis.strip()}"
    )
    return await _together_chat_async(
        client,
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=1024,
        temperature=0.2,
    )


async def _generate_detailed_async(
    client: httpx.AsyncClient,
    synthesis: str,
) -> Tuple[Optional[str], Optional[str]]:
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
    return await _together_chat_async(
        client,
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=4096,
        temperature=0.2,
    )


def summarize_chunk_with_context(
    chunk: str,
    chunk_index: int,
    num_chunks: int,
    prior_notes: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Synchronous single-chunk summary (tests / legacy); uses transcript-style prior string."""
    return asyncio.run(_summarize_chunk_standalone(chunk, chunk_index, num_chunks, prior_notes))


async def _summarize_chunk_standalone(
    chunk: str,
    chunk_index: int,
    num_chunks: int,
    prior: str,
) -> Tuple[Optional[str], Optional[str]]:
    async with httpx.AsyncClient(timeout=_together_timeout()) as client:
        return await _summarize_chunk_async(client, chunk, chunk_index, num_chunks, prior)


def reduce_notes(notes: List[str], max_chars: int) -> Tuple[str, Optional[str]]:
    """Synchronous merge (legacy); prefers one asyncio.run for the whole reduce tree."""

    async def _run() -> Tuple[str, Optional[str]]:
        sem = asyncio.Semaphore(SUMMARY_MAX_PARALLEL)
        limits = httpx.Limits(
            max_keepalive_connections=SUMMARY_MAX_PARALLEL,
            max_connections=SUMMARY_MAX_PARALLEL + 2,
        )
        async with httpx.AsyncClient(timeout=_together_timeout(), limits=limits) as client:
            return await _reduce_notes_async(client, notes, max_chars, sem)

    return asyncio.run(_run())


async def _summarize_transcript_async(
    transcript: str,
    progress: gr.Progress | None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    t0 = time.perf_counter()
    chunks = chunk_transcript(transcript, max_chars=SUMMARY_CHUNK_CHARS)
    if not chunks:
        return None, None, "Could not chunk transcript."

    sem = asyncio.Semaphore(SUMMARY_MAX_PARALLEL)
    limits = httpx.Limits(
        max_keepalive_connections=SUMMARY_MAX_PARALLEL + 4,
        max_connections=SUMMARY_MAX_PARALLEL + 8,
    )
    async with httpx.AsyncClient(timeout=_together_timeout(), limits=limits) as client:
        n = len(chunks)
        report_progress(
            progress,
            0.58,
            f"Summarizing {n} transcript part(s) (up to {SUMMARY_MAX_PARALLEL} parallel)…",
            logger,
        )

        async def _map_one(i: int, ch: str) -> Tuple[int, Optional[str], Optional[str]]:
            async with sem:
                prior = _prior_transcript_tail(chunks, i)
                note, err = await _summarize_chunk_async(client, ch, i, n, prior)
                return i, note, err

        map_results = await asyncio.gather(
            *(_map_one(i, ch) for i, ch in enumerate(chunks)),
            return_exceptions=True,
        )

        ordered_map: List[Tuple[int, Optional[str], Optional[str]]] = []
        for res in map_results:
            if not isinstance(res, tuple):
                return None, None, truncate_message(f"Chunk summarization failed: {res}")
            ordered_map.append(res)
        ordered_map.sort(key=lambda row: row[0])

        partial_notes: List[str] = []
        for _i, note, err in ordered_map:
            if err:
                return None, None, err
            if note:
                partial_notes.append(note)

        map_s = time.perf_counter() - t0
        logger.info(
            "Summarization map phase done in %.2fs (%s chunk(s), parallel=%s)",
            map_s,
            n,
            SUMMARY_MAX_PARALLEL,
        )

        if n == 1 and partial_notes:
            synthesis = partial_notes[0].strip()
            report_progress(progress, 0.88, "Single segment: using chunk summary as synthesis…", logger)
            reduce_s = 0.0
        else:
            report_progress(progress, 0.88, "Merging section summaries…", logger)
            t_reduce = time.perf_counter()
            synthesis, err = await _reduce_notes_async(
                client, partial_notes, max_chars=SUMMARY_REDUCE_CHARS, sem=sem
            )
            reduce_s = time.perf_counter() - t_reduce
            if err:
                return None, None, err
            if not synthesis.strip():
                return None, None, "Summarization produced an empty synthesis."
            logger.info("Summarization reduce phase done in %.2fs", reduce_s)

        report_progress(progress, 0.92, "Generating concise and detailed summaries…", logger)
        t_final = time.perf_counter()

        async def _concise() -> Tuple[Optional[str], Optional[str]]:
            async with sem:
                return await _generate_concise_async(client, synthesis)

        async def _detailed() -> Tuple[Optional[str], Optional[str]]:
            async with sem:
                return await _generate_detailed_async(client, synthesis)

        (concise, err_c), (detailed, err_d) = await asyncio.gather(_concise(), _detailed())
        final_s = time.perf_counter() - t_final
        logger.info("Final concise+detailed phase done in %.2fs", final_s)

        if err_c:
            return None, None, err_c
        if err_d:
            return concise, None, err_d

        total_s = time.perf_counter() - t0
        logger.info(
            "Summarization total %.2fs (map %.2fs, reduce %.2fs, final %.2fs)",
            total_s,
            map_s,
            reduce_s if n > 1 else 0.0,
            final_s,
        )
        return concise, detailed, None


def summarize_transcript(
    transcript: str,
    progress: gr.Progress | None = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Full pipeline: chunk → parallel per-chunk notes (with transcript overlap) → reduce →
    parallel concise + detailed.
    Returns (concise, detailed, error). If error is set, concise/detailed may be None.
    """
    t = (transcript or "").strip()
    if not t:
        return None, None, "Nothing to summarize (empty transcript)."

    cache_key = _transcript_cache_key(t)
    cached = _cache_get(cache_key)
    if cached is not None:
        concise, detailed = cached
        logger.info("Summary cache hit (sha256=%s…)", cache_key[:16])
        report_progress(progress, 1.0, "Summarization complete (cached)", logger)
        return concise, detailed, None

    report_progress(progress, 0.52, "Checking Together API configuration…", logger)
    if not _get_api_key():
        return None, None, (
            f"Missing {TOGETHER_API_KEY_ENV}. Summarization is disabled until the secret is set."
        )

    try:
        concise, detailed, err = asyncio.run(_summarize_transcript_async(t, progress))
    except RuntimeError as e:
        # Nested asyncio.run (unlikely in Gradio sync handler)
        logger.exception("asyncio.run failed in summarize_transcript")
        return None, None, truncate_message(f"Summarization runtime error: {e}")

    if err:
        return concise, detailed, err

    if concise is not None and detailed is not None:
        _cache_put(cache_key, concise, detailed)

    report_progress(progress, 1.0, "Summarization complete", logger)
    return concise, detailed, None
