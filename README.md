# FastYOLO

FastYOLO is a high-performance video object detection library powered by [Ultralytics YOLO v11](https://github.com/ultralytics/ultralytics).
It is designed to be fast, efficient, and easy to use.
FastYOLO maximizes GPU utilization through parallel video decoding and model inference:

1. Video decoding is performed asynchronously on CPU using FFmpeg
2. Model inference runs concurrently on GPU using PyTorch
3. A prefetch queue maintains a buffer of decoded frames ready for inference

This architecture eliminates the typical bottleneck where GPU processing waits for CPU video decoding, resulting in significantly higher throughput compared to sequential processing.

## Installation

First make sure ffmpeg is installed.

```bash
pip install fastyolo
```

## Usage

```python
import torch
from ultralytics import YOLO
from fastyolo import Video

# Initialize model as usual with Ultralytics
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt", task="detect").to(device)
dtype = next(model.parameters()).dtype

# Create video reader
video = Video(
    "traffic.mp4",
    # 640x640 is the default resolution for YOLO v11
    width=640,
    height=640,
    device=device,
    dtype=dtype
)

# Process video frames in batches
for batch in video:
    # Run inference
    results = model.predict(batch)
    
    # Process results
    for result in results:
        boxes = result.boxes 

```

## Todo 

- [ ] Optimize Yolo models including post processing
- [ ] Add support for other tasks
- [ ] Add GPU-accelerated video decoding using [torchcodec](https://github.com/pytorch/torchcodec) and other backends
