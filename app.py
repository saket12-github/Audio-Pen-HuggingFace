"""Gradio entrypoint: Audio Pen — transcription + Together AI summaries."""
from __future__ import annotations

import logging
import os

import gradio as gr

from summarization import summarize_transcript
from transcription import transcribe_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("audio_pen")


def process_audio(audio_path: str | None, progress: gr.Progress = gr.Progress()):
    """Returns: full transcript, concise summary, detailed summary, status."""
    progress(0.0, desc="Starting…")
    transcript, terr = transcribe_audio(audio_path, progress=progress)
    if terr:
        return "", "", "", terr
    assert transcript is not None

    concise, detailed, serr = summarize_transcript(transcript, progress=progress)
    if serr:
        return transcript, "", "", f"Transcription OK. Summarization: {serr}"

    progress(1.0, desc="Done")
    return transcript, concise or "", detailed or "", "Complete: transcript + summaries."


logger.info("Building Gradio UI (cwd=%s)", os.getcwd())

with gr.Blocks(title="Audio Pen") as demo:
    gr.Markdown(
        "# Audio Pen\n"
        "Upload audio → **transcript** → **concise** and **detailed** summaries (Together AI). "
        "Set the `TOGETHER_API_KEY` secret on Hugging Face Spaces."
    )
    audio_in = gr.Audio(
        type="filepath",
        label="Audio file",
    )
    go = gr.Button("Transcribe & summarize", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        tx = gr.Textbox(label="Full transcript", lines=14, interactive=False)
    with gr.Row():
        c = gr.Textbox(label="Concise summary", lines=8, interactive=False)
        d = gr.Textbox(label="Detailed summary", lines=14, interactive=False)

    go.click(
        fn=process_audio,
        inputs=audio_in,
        outputs=[tx, c, d, status],
    )


if __name__ == "__main__":
    logger.info("Launching locally")
    demo.launch()
