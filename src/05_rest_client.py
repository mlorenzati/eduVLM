#!/usr/bin/env python3
"""
05_rest_client.py — consume the eduVLM REST API with the standard OpenAI SDK.

The server (04_rest_api.py) speaks the OpenAI API, so any OpenAI client works:
just point base_url at the server. No custom protocol needed.

Usage:
    ./05_rest_client.py [image_path] [prompt]

Defaults to frames/frame_0000.jpg and a demo prompt.
"""

import base64
import mimetypes
import sys
from pathlib import Path

from openai import OpenAI

# Point the standard OpenAI client at our local server.
# api_key is required by the SDK but our server does not check it.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="eduVLM",  # placeholder, server ignores it
)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

image_path = sys.argv[1] if len(sys.argv) > 1 else "frames/frame_0000.jpg"
prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe what you see in one sentence."

if not Path(image_path).exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

# Encode the image as a base64 data URL (the standard way to inline media)
mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
with open(image_path, "rb") as f:
    data_url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }
]

print(f"Image:  {image_path}")
print(f"Prompt: {prompt}")
print()

# --- Non-streaming ---------------------------------------------------------
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=128,
)
print("Answer:", response.choices[0].message.content)
print()

# --- Streaming -------------------------------------------------------------
print("Streaming answer:")
stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
