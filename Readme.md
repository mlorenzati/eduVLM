# Edu VLM
Learning VLM by examples

## Component view
```
                  video.mp4
                      │
                      ▼
              ┌───────────────┐
              │ Video decoder │
              │    OpenCV     │
              └───────┬───────┘
                      │
                 sampled frames
                      │
             ┌────────▼────────┐
             │ Vision processor │
             └────────┬────────┘
                      │
                visual tokens
                      │
                      ▼
             ┌─────────────────┐
             │   VLM           │
             │ Qwen2.5-VL-3B   │
             └────────┬────────┘
                      │
                      ▼
                 text answer
```

## Pipeline view
```
                 IMAGE
                   │
                   ▼
          ┌─────────────────┐
          │ Vision Encoder  │
          └────────┬────────┘
                   │
             visual features
                   │
                   ▼
          ┌─────────────────┐
          │ Vision-Language │
          │     bridge      │
          └────────┬────────┘
                   │
                   ▼
        visual token embeddings
                   │
                   │
Question ──────────┤
                   ▼
          ┌─────────────────┐
          │ Language Model  │
          │   Transformer   │
          └────────┬────────┘
                   │
                   ▼
              text tokens
```

Video → frames
Frames → tensors
Tensors → vision encoder
Vision features → visual tokens
Visual tokens + text tokens
Multimodal transformer
Next-token prediction
Generated answer

When dealing with video
```
                 VIDEO
                   │
                   ▼
              sampling
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       frame 0  frame 1  frame 2
          │        │        │
          └────────┼────────┘
                   ▼
             vision encoder
                   │
                   ▼
              visual tokens
                   │
                   ▼
                Qwen
                   │
                   ▼
              text answer
```

## Setup
Environment
```
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

## Videos
```
cd videos
./getHughingFaceVids.sh 
```

## Installation
```
pip install -r requirements.txt
```

## Usage
Go to src and run the python examples by number

```
./01_extract_frames.py video_path
./02_image_tensor.py frames/frame_0000.jpg 
./03_image_question.py frames/frame_0000.jpg "how many people are on the scene"
```