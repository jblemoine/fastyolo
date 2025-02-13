from pathlib import Path

import httpx
import pytest
import torch
from ultralytics import YOLO

from fastyolo import Video
from fastyolo.video import VideoInfo

DATA_DIR = Path(__file__).parent / "data"


def download_video(url: str, output_path: Path) -> None:
    """Download a video from a URL to the specified path.

    Parameters
    ----------
    url : str
        URL of the video to download
    output_path : Path
        Path where the video will be saved

    Raises
    ------
    RuntimeError
        If the download fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        output_path.write_bytes(response.content)


@pytest.fixture
def video_path():
    url = "https://www.pexels.com/download/video/854671/"
    video_path = DATA_DIR / "video.mp4"

    if not video_path.exists():
        download_video(url=url, output_path=video_path)

    return video_path


def test_fastyolo(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt", task="detect").to(device)
    dtype = next(model.parameters()).dtype

    video = Video(video_path, width=640, height=640, device=device, dtype=dtype)
    for batch in video:
        results = model.predict(batch)
        assert len(results) == len(batch)


def test_VideoInfo(video_path):
    video_info = VideoInfo.from_path(video_path)
    assert video_info.width == 1920
    assert video_info.height == 1080
