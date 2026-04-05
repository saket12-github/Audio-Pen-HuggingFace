---
title: Audio Pen
emoji: 🎙️
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Transcribe audio with Whisper and summarize with Together AI
---

# Audio Pen

Upload an audio file, get a **full transcript** plus **concise** and **detailed** summaries. Built for [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview) (CPU-friendly Whisper via `faster-whisper`; summarization via [Together AI](https://www.together.ai/)).

## Secrets (required for summaries)

In the Space: **Settings → Secrets and variables → New secret**

| Name | Value |
|------|--------|
| `TOGETHER_API_KEY` | Your Together API key |

Never commit API keys. The app reads only `os.environ["TOGETHER_API_KEY"]`.

Transcription still runs if the key is missing; summarization will report a clear error.

## Optional environment variables

| Variable | Purpose |
|----------|---------|
| `TOGETHER_MODEL` | Override default LLM (default: `meta-llama/Llama-3.3-70B-Instruct-Turbo`) |
| `WHISPER_MODEL_SIZE` | Override Whisper checkpoint (default: `distil-large-v3`) |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | e.g. `cpu` + `int8` on Spaces |
| `SUMMARY_CHUNK_CHARS` | Transcript chunk size for map-reduce (~chars per API call) |
| `MAX_AUDIO_MB` | Max upload size (default: 100) |

## Hardware

Whisper on **free CPU** can be slow for long files. For heavier use, pick **CPU upgrade** (or GPU) in Space settings.

## Layout

- `app.py` — Gradio UI (`demo` is the `Blocks` instance Spaces expects)
- `transcription.py` — validation + `faster-whisper`
- `summarization.py` — Together chat completions, chunk → merge → concise + detailed
- `config.py` — env-driven settings
- `apt.txt` — installs **ffmpeg** for decoding common audio formats
