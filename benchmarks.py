# Benchmark with NVTX annotations for GPU profiling 
import os
import time
from functools import wraps
from pathlib import Path

import torch
from ultralytics import YOLO

from fastyolo import Video

DATA_DIR = Path(__file__).parent / "data"


def timeit_function(func, number, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)

    start = time.time()
    for _ in range(number):
        wrapper()
    end = time.time()
    average_time = (end - start) / number
    print(
        f"Average time for {func.__name__} over {number} runs: {average_time:.4f} seconds"
    )

def run_fastyolo(model: YOLO, video_path: Path):

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    video =Video(
        video_path, width=640, height=640, device=device, dtype=dtype
    )
    for batch in video:
        model.predict(batch, device=device, verbose=False, half=dtype == torch.float16)

def run_ultralytics(model: YOLO, video_path: Path):
    for _ in model.predict(video_path, stream=True, device=next(model.parameters()).device, verbose=False):
        pass

@torch.no_grad()
def run_no_postprocess(model: torch.nn.Module, video_path: Path):
    model.eval()
    video = Video( video_path, width=640, height=640, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    for batch in video:
        model(batch)


if __name__ == "__main__":
    model = YOLO("yolo11n.pt", task="detect")
    model.fuse()
    model.to("cuda")
    model.half()

    video_path = DATA_DIR / "video_1080p.mp4"

    timeit_function(run_fastyolo, number=1, model=model, video_path=video_path)
    # timeit_function(run_ultralytics, number=1, model=model, video_path=video_path)

    timeit_function(run_no_postprocess, number=1, model=model.model, video_path=video_path)

