from pathlib import Path

import torch
from ultralytics import YOLO

from fastyolo import Video

DATA_DIR = Path(__file__).parent / "data"


def test_fastyolo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt", task="detect").to(device)
    dtype = next(model.parameters()).dtype
    video = Video(
        DATA_DIR / "traffic.mp4", width=640, height=640, device=device, dtype=dtype
    )
    for batch in video:
        results = model.predict(batch)
        assert len(results) == len(batch)
