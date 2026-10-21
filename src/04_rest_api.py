#!/usr/bin/env python3
"""
04_rest_api.py — OpenAI-compatible REST API for Qwen2.5-VL-3B.

Exposes the VLM so it can be consumed externally by any OpenAI-compatible
client (openai SDK, curl, LangChain, etc).

Endpoints:
    GET  /health                liveness + device info
    GET  /v1/models             list served models (OpenAI format)
    POST /v1/chat/completions   chat completions (OpenAI format, stream + non-stream)

Media is passed inside the standard OpenAI message content:
    {"type": "image_url", "image_url": {"url": "<data URL | http(s) URL | local path>"}}
Videos use the Qwen-style extension (same as vLLM):
    {"type": "video", "video": "<data URL | http(s) URL | local path>"}

Run:
    ./04_rest_api.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import base64
import mimetypes
import os
import tempfile
import threading
import time
import uuid
from typing import Annotated, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_FPS = 0.5  # frames per second sampled from video, same as 03_media_question.py

# ---------------------------------------------------------------------------
# OpenAI-compatible request schemas (the subset we support)
# ---------------------------------------------------------------------------


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = None


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


class ContentPartVideo(BaseModel):
    type: Literal["video"]
    video: str  # data URL, http(s) URL or server-local path


ContentPart = Annotated[
    Union[ContentPartText, ContentPartImage, ContentPartVideo],
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str = MODEL
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Model loading (same pattern as 03_media_question.py, but done once at boot)
# ---------------------------------------------------------------------------

print(f"Loading model {MODEL}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model loaded. Device: {device}")

app = FastAPI(title="eduVLM", description="Qwen2.5-VL-3B OpenAI-compatible API")


# ---------------------------------------------------------------------------
# Media helpers: turn OpenAI content parts into Qwen vision content dicts
# ---------------------------------------------------------------------------


def decode_data_url(data_url: str) -> str:
    """Decode a base64 data URL into a temp file and return its path."""
    try:
        header, payload = data_url.split(",", 1)
    except ValueError:
        raise HTTPException(400, "Malformed data URL")
    mime = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    ext = mimetypes.guess_extension(mime) or ".bin"
    try:
        raw = base64.b64decode(payload)
    except Exception:
        raise HTTPException(400, "Invalid base64 payload in data URL")
    fd, path = tempfile.mkstemp(suffix=ext, prefix="eduVLM_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def to_qwen_content(part) -> dict:
    """Convert one OpenAI content part into a Qwen-VL content dict."""
    if part.type == "text":
        return {"type": "text", "text": part.text}
    if part.type == "image_url":
        url = part.image_url.url
        if url.startswith("data:"):
            return {"type": "image", "image": decode_data_url(url)}
        return {"type": "image", "image": url}  # http(s) URL or local path
    if part.type == "video":
        source = part.video
        if source.startswith("data:"):
            source = decode_data_url(source)
        return {"type": "video", "video": source, "fps": VIDEO_FPS}
    raise HTTPException(400, f"Unsupported content part type: {part.type}")


def build_qwen_messages(messages) -> List[dict]:
    """Map OpenAI chat messages to Qwen chat-template messages."""
    qwen_messages = []
    for message in messages:
        if message.role not in {"system", "user", "assistant"}:
            raise HTTPException(400, f"Unsupported role: {message.role}")
        if isinstance(message.content, str):
            content = [{"type": "text", "text": message.content}]
        else:
            content = [to_qwen_content(part) for part in message.content]
        qwen_messages.append({"role": message.role, "content": content})
    return qwen_messages

def generate_answer(qwen_messages: List[dict], request: ChatCompletionRequest) -> str:
    """Run the full VLM pipeline: template -> vision info -> tensors -> generate."""
    text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(qwen_messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 512,
            do_sample=request.temperature > 0,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    generated_ids = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, output_ids)
    ]

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def cleanup_temp_files(qwen_messages: List[dict]) -> None:
    """Remove temp files created from base64 data URLs for this request."""
    for message in qwen_messages:
        for part in message["content"]:
            source = part.get("image") or part.get("video")
            if isinstance(source, str) and source.startswith(tempfile.gettempdir()):
                try:
                    os.remove(source)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL, "object": "model", "created": 0, "owned_by": "eduVLM"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    qwen_messages = build_qwen_messages(request.messages)

    if not request.stream:
        try:
            answer = generate_answer(qwen_messages, request)
        finally:
            cleanup_temp_files(qwen_messages)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
        }

    # Streaming: real token-by-token SSE using TextIteratorStreamer
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    def run_generation():
        try:
            text = processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(qwen_messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 512,
                    do_sample=request.temperature > 0,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    streamer=streamer,
                )
        finally:
            cleanup_temp_files(qwen_messages)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    def sse_chunks():
        for token in streamer:
            if not token:
                continue
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }
            yield f"data: {chunk}\n\n"
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {final}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_chunks(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="eduVLM OpenAI-compatible API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

