import subprocess
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterator

import numpy as np
import torch

from fastyolo.utils import rgb_to_hex

PathLike = Path | str


def prefetch_iter(iter: Iterator, queue_size: int):
    def producer(iter: Iterator, queue: Queue):
        for item in iter:
            queue.put(item)
        queue.put(None)  # Sentinel value to signal end of iteration

    queue: Queue = Queue(maxsize=queue_size)
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
    def from_path(cls, video_path: PathLike):
        """
        Extract video info using ffprobe
        """
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract video info: {result.stderr}")
        width, height, total_frames = result.stdout.split(",")
        return cls(width=int(width), height=int(height), total_frames=int(total_frames))


class Video:
    def __init__(
        self,
        video_path: PathLike,
        device: str,
        dtype: torch.dtype,
        backend: str = "ffmpeg",
        width: int | None = None,
        height: int | None = None,
        letterbox: bool = True,
        letterbox_color: tuple[int, int, int] = (114, 114, 114),
        batch_size: int = 16,
        prefetch: bool = True,
        prefetch_queue_size: int | None = None,
    ):
        self.video_path = Path(video_path)
        # self.info = VideoInfo.from_path(self.video_path)
        self.backend = backend
        # self.source_width = self.info.width
        # self.source_height = self.info.height
        self.target_width = width or self.source_width
        self.target_height = height or self.source_height
        self.letterbox = letterbox
        self.letterbox_color = letterbox_color
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.prefetch_queue_size = prefetch_queue_size or batch_size * 2
        self.device = device
        self.dtype = dtype

    def frames(self) -> Iterator[np.ndarray]:
        command = [
            "ffmpeg",
            "-i",
            str(self.video_path),
            "-y",
        ]

        if self.letterbox:
            command.extend(
                [
                    "-vf",
                    f"scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=decrease,pad={self.target_width}:{self.target_height}:(ow-iw)/2:(oh-ih)/2:color={rgb_to_hex(*self.letterbox_color)}",
                ]
            )

        command.extend(
            ["-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
        )

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if process.stdout is None:
            raise RuntimeError("Failed to open ffmpeg stdout pipe")

        if process.returncode is not None and process.returncode != 0:
            raise RuntimeError(
                f"Failed to start ffmpeg process: {process.stderr.read().decode() if process.stderr else ''}"
            )

        while True:
            raw_frame = process.stdout.read(self.target_width * self.target_height * 3)
            if len(raw_frame) != self.target_width * self.target_height * 3:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (self.target_height, self.target_width, 3)
            )
            yield frame

        process.stdout.close()
        process.wait()

    @torch.inference_mode()
    def preprocess(self, batch: list[np.ndarray]) -> torch.Tensor:
        tensors = [torch.from_numpy(frame) for frame in batch]
        tensor = torch.stack(tensors)
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.to(self.device, dtype=self.dtype, non_blocking=True)
        tensor = tensor / 255.0
        return tensor

    def decode(self) -> Iterator[torch.Tensor]:
        frames = self.frames()
        if self.prefetch:
            frames = prefetch_iter(frames, self.prefetch_queue_size)
        batches = make_batches(frames, self.batch_size)
        for batch in batches:
            yield self.preprocess(batch)

    def __iter__(self):
        return self.decode()
