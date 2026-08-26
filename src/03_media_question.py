#!/usr/bin/env python3

import torch
import sys
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info

if len(sys.argv) <= 2 or len(sys.argv) > 3:
    print(f"{sys.argv[0]} media_path prompt")
    sys.exit(1)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
media_path = sys.argv[1]
prompt = sys.argv[2]

media_ext = Path(media_path).suffix.lower()

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

video_extensions = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"
}

if media_ext in image_extensions:
    media_content = {
        "type": "image",
        "image": media_path,
    }
elif media_ext in video_extensions:
    media_content = {
        "type": "video",
        "video": media_path,
        "fps": 0.5,
    }
else:
    print(f"Unsupported media type: {media_ext}")
    sys.exit(1)


print(f"Loading model {MODEL}...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL)

print("Model loaded.")

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

messages = [
    {
        "role": "user",
        "content": [
            media_content,
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


image_inputs, video_inputs = process_vision_info(
    messages
)


inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


inputs = inputs.to("cuda")


print()
print("Input tensors:")

for name, value in inputs.items():
    if hasattr(value, "shape"):
        print(
            f"  {name:20s}",
            value.shape,
            value.dtype,
        )

print(f"Running inference for:\Media {media_path}\nPrompt: {prompt}")


with torch.inference_mode():

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
    )


generated_ids = [
    output[len(input_ids):]
    for input_ids, output in zip(
        inputs.input_ids,
        output_ids
    )
]


answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print()
print("========== ANSWERs ==========")
print(f"Number of answers: {len(answer)}")
for count, ans in enumerate(answer, start=1):
    print(f"Answer {count}: {ans}")
print("=============================")