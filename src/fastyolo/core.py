import torch
from pathlib import Path
from enum import Enum
import subprocess
import numpy as np
from fastyolo.utils import rgb_to_hex
from typing import Iterator
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from ultralytics.utils.downloads import attempt_download_asset
from functools import cached_property

 
MODEL_DIR = Path.home() / ".cache" / "fastyolo"

class ModelName(str, Enum):
    YOLO11N = "yolo11n.pt"
    YOLO11S = "yolo11s.pt"
    YOLO11M = "yolo11m.pt"
    YOLO11L = "yolo11l.pt"
    YOLO11X = "yolo11x.pt"

def get_or_download_model(model_name: ModelName, release: str) -> Path:

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        model_path = attempt_download_asset(model_path, repo="ultralytics/assets", release=release)
    return Path(model_path)

model_default_image_size = {
    ModelName.YOLO11N: (640, 640),
    ModelName.YOLO11S: (640, 640),
    ModelName.YOLO11M: (640, 640),
    ModelName.YOLO11L: (640, 640),
    ModelName.YOLO11X: (640, 640),
}

class Predictor:
    def __init__(self, model_name: str, release: str = "v8.3.0"):
        self.model_name = model_name
        self.release = release
        self.model_path = get_or_download_model(model_name, release=self.release)
        self.model = torch.load(self.model_path)["model"]
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def preprocess(self, batch: list[np.ndarray]) -> torch.Tensor:
        batch = [torch.from_numpy(frame) for frame in batch]
        batch = torch.stack(batch)
        batch = batch.permute(0, 3, 1, 2)
        batch = batch.to(self.model.device, dtype=self.model.dtype, non_blocking=True)
        batch = batch / 255.0
        return batch
        
    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)
    
def prefetch_iter(iter: Iterator, queue_size: int):
    def producer(iter: Iterator, queue: Queue):
        for item in iter:
            queue.put(item)
        queue.put(None)  # Sentinel value to signal end of iteration

    queue = Queue(maxsize=queue_size)
    thread = Thread(target=producer, args=(iter, queue), daemon=True)
    thread.start()

    while True:
        item = queue.get()
        if item is None:
            break
        yield item

def make_batches(iter: Iterator, batch_size: int):
    batch = []
    for item in iter:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class VideoInfo:
    width: int
    height: int
    total_frames: int


@classmethod
def from_path(cls, video_path: str):
    """
    Extract video info using ffprobe
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract video info: {result.stderr}")
    width, height, total_frames = result.stdout.split(",")
    return cls(width=int(width), height=int(height), total_frames=int(total_frames))



class Video:
    def __init__(self, video_path: str, backend: str = "ffmpeg", prefetch: bool = False,
                  prefetch_queue_size: int | None = None, width: int | None = None, height: int | None = None,
                    letterbox: bool = True, letterbox_color : tuple[int, int, int] = (114, 114, 114)):
        self.video_path = video_path
        self.backend = backend
        self.prefetch = prefetch
        self.prefetch_queue_size = prefetch_queue_size
        self.width = width
        self.height = height
        self.letterbox = letterbox
        self.letterbox_color = letterbox_color

    @cached_property
    def info(self) -> VideoInfo:
        return VideoInfo.from_path(self.video_path)

    def decode(self) -> Iterator[np.ndarray]:
        command = [
            "ffmpeg",
            "-i", self.video_path,
            "-y",
        ]

        if self.letterbox:
            command.extend([
                "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color={rgb_to_hex(*self.letterbox_color)}",
            ])

        command.extend([
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"])
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            raise RuntimeError(f"Failed to start ffmpeg process: {self.process.stderr.read()}")

        while True:
            raw_frame = process.stdout.read(self.width * self.height * 3)
            if len(raw_frame) != self.width * self.height * 3:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
            yield frame

        process.stdout.close()
        process.wait()

    def stream(self) -> Iterator[np.ndarray]:
        frames = self.decode()
        if self.prefetch:
            frames = prefetch_iter(frames, self.prefetch_queue_size)
        return make_batches(frames, self.batch_size)
    
    def __iter__(self):
        return self.stream()
    
class FastYolo:
    def __init__(self, model_name: str | ModelName):
        self.model_name = ModelName(model_name)
        self.model = Predictor(self.model_name)


    def detect_video(self, 
                     video_path: str, 
                     height: int | None = None, 
                     width: int | None = None,
                     letterbox: bool = True,
                     letterbox_color: tuple[int, int, int] = (114, 114, 114),
                     batch_size: int = 16,
                     prefetch: bool = False,
                     prefetch_queue_size: int | None = None) -> Iterator[torch.Tensor]:
        
        height = height or model_default_image_size[self.model_name][0]
        width = width or model_default_image_size[self.model_name][1]

        if prefetch:
            prefetch_queue_size = prefetch_queue_size or batch_size * 2

        video = Video(video_path, width=width, height=height, letterbox=letterbox, letterbox_color=letterbox_color, prefetch=prefetch, prefetch_queue_size=prefetch_queue_size, batch_size=batch_size)

        for batch in video:
            tensor = self.model.preprocess(batch)
            results = self.model.predict(tensor)
            for result in results:
                yield result

