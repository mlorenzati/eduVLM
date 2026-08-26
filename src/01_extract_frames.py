#!/usr/bin/env python3

import cv2
import os
import sys

if len(sys.argv) <= 1:
    print(f"{sys.argv[0]} video_path frames_path")
    sys.exit(1)

VIDEO = sys.argv[1]

output_dir = "../frames"
if len(sys.argv) >= 3:
    output_dir = sys.argv[2]

FPS_SAMPLE = 1.0

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO}")


video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frame_count / video_fps

print(f"Video FPS:       {video_fps:.2f}")
print(f"Frame count:     {frame_count}")
print(f"Duration:        {duration:.2f} seconds")
print(f"Sampling FPS:    {FPS_SAMPLE}")


frame_interval = video_fps / FPS_SAMPLE

next_frame = 0
index = 0

while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_frame))

    ret, frame = cap.read()

    if not ret:
        break

    filename = f"{output_dir}/frame_{index:04d}.jpg"

    cv2.imwrite(filename, frame)

    print(f"Saved {filename}")

    index += 1
    next_frame += frame_interval


cap.release()

print()
print(f"Extracted {index} frames to {output_dir}")