#!/usr/bin/env python3

from PIL import Image
import torch
import sys
from torchvision import transforms

if len(sys.argv) <= 1:
    print(f"{sys.argv[0]} image_path")
    sys.exit(1)

image_file=sys.argv[1]
image = Image.open(f"{image_file}")

print("Original image:")
print("  name:", image_file)
print("  size:", image.size)
print("  mode:", image.mode)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tensor = transform(image)

print()
print("Tensor:")
print("  shape:", tensor.shape)
print("  dtype:", tensor.dtype)
print("  device:", tensor.device)
print("  min:", tensor.min().item())
print("  max:", tensor.max().item())