from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

import config
from detector import HeuristicResult, Stage1Detector


@dataclass
class VideoRunner:
    video_path: str
    loop_start_sec: float = 0.0
    loop_end_sec: Optional[float] = None
    frame_skip: int = 0

    def run(
        self,
        detector: Stage1Detector,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, HeuristicResult], None, None]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        loop_start_frame = max(0, int(round(self.loop_start_sec * fps)))
        loop_end_frame = self._resolve_loop_end_frame(fps, loop_start_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start_frame)
        logical_frame = loop_start_frame
        emitted = 0

        try:
            while max_frames is None or emitted < max_frames:
                if loop_end_frame is not None and logical_frame >= loop_end_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start_frame)
                    logical_frame = loop_start_frame

                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start_frame)
                    logical_frame = loop_start_frame
                    continue

                should_process = ((logical_frame - loop_start_frame) % (self.frame_skip + 1)) == 0
                logical_frame += 1
                if not should_process:
                    continue

                result = detector.process_frame(frame)
                emitted += 1
                yield frame, result
        finally:
            cap.release()

    def _resolve_loop_end_frame(self, fps: float, loop_start_frame: int) -> Optional[int]:
        if self.loop_end_sec is None:
            return None
        if self.loop_end_sec <= self.loop_start_sec:
            return None
        return max(loop_start_frame + 1, int(round(self.loop_end_sec * fps)))


def _probe_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0
    finally:
        cap.release()


def build_stage1_detector() -> Stage1Detector:
    fps = _probe_video_fps(config.video_path)
    return Stage1Detector(
        visibility_threshold=config.visibility_threshold,
        heuristics_window_seconds=config.heuristics_window_seconds,
        min_heuristics=config.min_heuristics,
        fps=fps,
        pose_model_path=config.pose_landmarker_model_path,
    )


def run_stage1(
    max_frames: Optional[int] = None,
    frame_skip: int = 0,
) -> Generator[Tuple[np.ndarray, HeuristicResult], None, None]:
    detector = build_stage1_detector()
    runner = VideoRunner(
        video_path=config.video_path,
        loop_start_sec=config.demo_loop_start,
        loop_end_sec=config.demo_loop_end,
        frame_skip=frame_skip,
    )
    return runner.run(detector=detector, max_frames=max_frames)


def main() -> None:
    for _, result in run_stage1(max_frames=300):
        if result.flag:
            fired = ", ".join(result.fired_heuristics)
            print(f"[Stage1 FLAG] confidence={result.confidence:.2f} fired={fired}")


if __name__ == "__main__":
    main()
