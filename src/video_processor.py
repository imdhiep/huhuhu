"""
Video Processor: Extract frames from MP4 videos for VLM inference.
Supports multiple sampling strategies: uniform, fps-based, keyframes.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process MP4 videos and extract frames for VLM input."""

    def __init__(
        self,
        target_fps: int = 3,
        max_frames: int = 32,
        resolution: int = 384,
        sampling_strategy: str = "uniform"
    ):
        """
        Initialize video processor.

        Args:
            target_fps: Target frames per second for extraction
            max_frames: Maximum number of frames to extract
            resolution: Resize frames to this size (square)
            sampling_strategy: "uniform", "fps", or "keyframe"
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.resolution = resolution
        self.sampling_strategy = sampling_strategy

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video file.

        Args:
            video_path: Path to MP4 video file

        Returns:
            List of frames as numpy arrays (RGB)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0

            logger.info(f"Video: {video_path.name}")
            logger.info(f"  Total frames: {total_frames}")
            logger.info(f"  FPS: {video_fps:.2f}")
            logger.info(f"  Duration: {duration:.2f}s")

            # Select frames based on strategy
            frame_indices = self._select_frames(total_frames, video_fps)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize
                    frame_resized = cv2.resize(
                        frame_rgb,
                        (self.resolution, self.resolution),
                        interpolation=cv2.INTER_AREA
                    )
                    frames.append(frame_resized)

            logger.info(f"Extracted {len(frames)} frames using {self.sampling_strategy} strategy")
            return frames

        finally:
            cap.release()

    def _select_frames(self, total_frames: int, video_fps: float) -> List[int]:
        """Select frame indices based on sampling strategy."""
        if total_frames <= self.max_frames:
            # Use all frames if fewer than max
            return list(range(total_frames))

        if self.sampling_strategy == "uniform":
            # Uniform sampling across the video
            step = max(1, total_frames // self.max_frames)
            return list(range(0, total_frames, step))[:self.max_frames]

        elif self.sampling_strategy == "fps":
            # Sample at target FPS
            frame_interval = max(1, int(video_fps / self.target_fps))
            return list(range(0, total_frames, frame_interval))[:self.max_frames]

        elif self.sampling_strategy == "keyframe":
            # Simple keyframe detection (I-frames are typically keyframes)
            # For simplicity, we'll use uniform sampling but could be enhanced
            step = max(1, total_frames // self.max_frames)
            return list(range(0, total_frames, step))[:self.max_frames]

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def frames_to_video_tensor(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Convert list of frames to video tensor format.

        Args:
            frames: List of RGB frames

        Returns:
            Video tensor of shape (T, H, W, C)
        """
        if not frames:
            raise ValueError("No frames to convert")

        video_tensor = np.stack(frames, axis=0)
        logger.info(f"Video tensor shape: {video_tensor.shape}")
        return video_tensor

    def get_frame_timestamps(
        self,
        total_frames: int,
        video_fps: float,
        frame_indices: List[int]
    ) -> List[float]:
        """Get timestamps for extracted frames."""
        return [idx / video_fps for idx in frame_indices]
