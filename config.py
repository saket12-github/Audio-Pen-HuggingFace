"""Central configuration for Audio Pen (HF Spaces–friendly defaults)."""
import os

# Whisper (faster-whisper)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "distil-large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
# Lower beam speeds up CPU inference; raise (e.g. 5) for higher quality on GPU.
_default_beam = "1" if WHISPER_DEVICE.lower() == "cpu" else "5"
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", _default_beam))
WHISPER_VAD_FILTER = os.environ.get("WHISPER_VAD_FILTER", "true").lower() in (
    "1",
    "true",
    "yes",
)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = os.environ.get(
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true"
).lower() in ("1", "true", "yes")

# Audio limits
MAX_AUDIO_MB = int(os.environ.get("MAX_AUDIO_MB", "100"))
SUPPORTED_FORMATS = frozenset(
    x.strip().lower().lstrip(".")
    for x in os.environ.get(
        "SUPPORTED_AUDIO_FORMATS",
        "mp3,wav,webm,m4a,ogg,flac,opus",
    ).split(",")
    if x.strip()
)

# Together AI (summarization)
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
TOGETHER_API_URL = os.environ.get(
    "TOGETHER_API_URL", "https://api.together.xyz/v1/chat/completions"
)
TOGETHER_MODEL = os.environ.get(
    "TOGETHER_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

# Chunking: character budget per LLM call (prompt + chunk + output)
SUMMARY_CHUNK_CHARS = int(os.environ.get("SUMMARY_CHUNK_CHARS", "9000"))
SUMMARY_REDUCE_CHARS = int(os.environ.get("SUMMARY_REDUCE_CHARS", "12000"))
CONTEXT_TAIL_CHARS = int(os.environ.get("SUMMARY_CONTEXT_TAIL_CHARS", "1200"))

# HTTP
TOGETHER_TIMEOUT_S = float(os.environ.get("TOGETHER_TIMEOUT_S", "120"))
TOGETHER_CONNECT_TIMEOUT_S = float(os.environ.get("TOGETHER_CONNECT_TIMEOUT_S", "15"))
TOGETHER_MAX_RETRIES = int(os.environ.get("TOGETHER_MAX_RETRIES", "2"))

# Rolling context cap (chars) for map step — avoids unbounded memory on long transcripts
SUMMARY_ROLLING_MAX_CHARS = int(os.environ.get("SUMMARY_ROLLING_MAX_CHARS", "24000"))
