from pathlib import Path

import torch
from ultralytics import YOLO

from fastyolo import Video
from fastyolo.video import VideoInfo

DATA_DIR = Path(__file__).parent / "data"

video_path = DATA_DIR / "traffic.mp4"


def test_fastyolo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt", task="detect").to(device)
    dtype = next(model.parameters()).dtype
    video = Video(
        video_path, width=640, height=640, device=device, dtype=dtype
    )
    for batch in video:
        results = model.predict(batch)
        assert len(results) == len(batch)

def test_VideoInfo():
    video_info = VideoInfo.from_path(video_path)
    assert video_info.width == 1920
    assert video_info.height == 1080
