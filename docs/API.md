# eduVLM REST API contract

OpenAI-compatible API served by `src/04_rest_api.py` (Qwen2.5-VL-3B-Instruct).
Any OpenAI-standard client can consume it by pointing `base_url` at the server.

- **Base URL:** `http://192.168.1.37:8000`
- **Auth:** none (the `api_key` field is accepted but ignored — send any value)
- **Machine-readable spec:** `GET /openapi.json` (interactive docs at `/docs`)

---

## Endpoints

| Method | Path                    | Description                                  |
|--------|-------------------------|----------------------------------------------|
| GET    | `/health`               | Liveness, model, device and GPU info         |
| GET    | `/v1/models`            | List served models (OpenAI format)           |
| POST   | `/v1/chat/completions`  | Chat completions (stream and non-stream)     |

### GET /health
```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3070 Laptop GPU"
}
```

### GET /v1/models
```json
{
  "object": "list",
  "data": [
    {"id": "Qwen/Qwen2.5-VL-3B-Instruct", "object": "model", "created": 0, "owned_by": "eduVLM"}
  ]
}
```

---

## POST /v1/chat/completions

### Request body
| Field         | Type    | Default                       | Notes                        |
|---------------|---------|-------------------------------|------------------------------|
| `model`       | string  | `Qwen/Qwen2.5-VL-3B-Instruct` | Must match the served id     |
| `messages`    | array   | required                      | OpenAI chat messages         |
| `max_tokens`  | int     | `512`                         | Max new tokens to generate   |
| `temperature` | float   | `0.7`                         | `0` = greedy decoding        |
| `top_p`       | float   | `0.9`                         | Nucleus sampling             |
| `stream`      | bool    | `false`                       | `true` = SSE token streaming |

### Message content parts
`content` is either a plain string or an array of typed parts:

| Part                                                     | Media source                                        |
|----------------------------------------------------------|-----------------------------------------------------|
| `{"type": "text", "text": "..."}`                         | —                                                   |
| `{"type": "image_url", "image_url": {"url": "<source>"}}` | base64 data URL, http(s) URL, or server-local path  |
| `{"type": "video", "video": "<source>"}`                  | base64 data URL, http(s) URL, or server-local path  |

Notes:
- **base64 data URL** (recommended for remote consumers): `data:image/jpeg;base64,<payload>` — the server decodes it to a temp file. Same for video: `data:video/mp4;base64,<payload>`.
- **http(s) URL**: the server downloads the media.
- **server-local path**: e.g. `/home/marcelo/Dev/ai/eduVLM/frames/frame_0000.jpg` — only for files already on the GPU machine.
- Video sampling is fixed at **0.5 fps**.
- Roles supported: `system`, `user`, `assistant`.

### Non-streaming response (`stream: false`)
```json
{
  "id": "chatcmpl-5b1f5f9b3bb0",
  "object": "chat.completion",
  "created": 1792617706,
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "There are three people visible in the scene."},
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming response (`stream: true`)
Server-Sent Events, one JSON chunk per token, terminated by `data: [DONE]`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1792617706,"model":"Qwen/Qwen2.5-VL-3B-Instruct","choices":[{"index":0,"delta":{"content":"There"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1792617706,"model":"Qwen/Qwen2.5-VL-3B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Errors
- `400` with `{"detail": "..."}` — malformed data URL, unsupported role or content part.
- `422` — request body validation errors (FastAPI/Pydantic format).

---

## Examples

### curl (image as base64 data URL)
```bash
B64=$(base64 -w0 image.jpg)
curl http://192.168.1.37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen2.5-VL-3B-Instruct\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$B64\"}},
        {\"type\": \"text\", \"text\": \"What do you see?\"}
      ]
    }],
    \"max_tokens\": 128
  }"
```

### curl (video on the server filesystem)
```bash
curl http://192.168.1.37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "video", "video": "/home/marcelo/Dev/ai/eduVLM/videos/video.mp4"},
        {"type": "text", "text": "What happens in this video?"}
      ]
    }],
    "max_tokens": 128
  }'
```

### Python — standard `openai` package
```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.37:8000/v1", api_key="eduVLM")

# non-streaming
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(response.choices[0].message.content)

# streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Python — image via base64 data URL
```python
import base64, mimetypes
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.37:8000/v1", api_key="eduVLM")
mime = mimetypes.guess_type("image.jpg")[0] or "image/jpeg"
b64 = base64.b64encode(Path("image.jpg").read_bytes()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": "Describe the scene."},
        ],
    }],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

---

## Limitations
- One request at a time (single GPU; concurrent requests are serialized by the model).
- `usage` token counts are not reported yet.
- No API key enforcement — do not expose beyond the LAN.
- Video uses the Qwen/vLLM-style `{"type": "video"}` part (not yet in the official OpenAI spec).

