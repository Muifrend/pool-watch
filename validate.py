from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import cv2

from detector import HEURISTIC_KEYS, STATUS_INDETERMINATE, Stage1Detector
from main import VideoRunner


def _probe_video_metadata(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    effective_fps = fps if fps and fps > 0 else 30.0
    total_frames = int(frame_count) if frame_count and frame_count > 0 else 0
    if total_frames <= 0:
        raise RuntimeError(
            "Unable to determine total frame count from video metadata; "
            "validation runner requires a finite frame count."
        )

    return effective_fps, total_frames


def _print_summary(
    total_frames: int,
    status_counts: Counter,
    confidences: list[float],
    heuristic_pass_counts: Dict[str, int],
    heuristic_evaluable_counts: Dict[str, int],
) -> None:
    print("Validation Summary")
    print(f"Total frames processed: {total_frames}")

    print("Status counts:")
    for status_name, count in sorted(status_counts.items()):
        print(f"  {status_name}: {count}")

    if confidences:
        print(
            "Confidence range: "
            f"min={min(confidences):.4f}, max={max(confidences):.4f}"
        )
    else:
        print("Confidence range: n/a")

    heuristic_rates: Dict[str, float] = {}
    for key in HEURISTIC_KEYS:
        evaluable = heuristic_evaluable_counts[key]
        passed = heuristic_pass_counts[key]
        rate = (passed / evaluable) if evaluable > 0 else 0.0
        heuristic_rates[key] = rate

    highest = max(HEURISTIC_KEYS, key=lambda k: heuristic_rates[k])
    lowest = min(HEURISTIC_KEYS, key=lambda k: heuristic_rates[k])

    print(
        "Highest pass-rate heuristic: "
        f"{highest} ({heuristic_rates[highest] * 100:.2f}% | "
        f"{heuristic_pass_counts[highest]}/{heuristic_evaluable_counts[highest]})"
    )
    print(
        "Lowest pass-rate heuristic: "
        f"{lowest} ({heuristic_rates[lowest] * 100:.2f}% | "
        f"{heuristic_pass_counts[lowest]}/{heuristic_evaluable_counts[lowest]})"
    )


def run_validation(video_path: str, output_csv: str) -> None:
    fps, total_frames = _probe_video_metadata(video_path)

    detector = Stage1Detector(fps=fps)
    runner = VideoRunner(
        video_path=video_path,
        loop_start_sec=0.0,
        loop_end_sec=None,
        frame_skip=0,
    )

    status_counts: Counter = Counter()
    confidences: list[float] = []
    heuristic_pass_counts = {key: 0 for key in HEURISTIC_KEYS}
    heuristic_evaluable_counts = {key: 0 for key in HEURISTIC_KEYS}

    output_path = Path(output_csv)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "frame_number",
            "timestamp_seconds",
            "status",
            "confidence",
            *HEURISTIC_KEYS,
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for frame_number, (_, result) in enumerate(runner.run(detector, max_frames=total_frames)):
            timestamp_seconds = frame_number / fps
            row = {
                "frame_number": frame_number,
                "timestamp_seconds": f"{timestamp_seconds:.6f}",
                "status": result.status,
                "confidence": f"{result.confidence:.6f}",
            }

            for key in HEURISTIC_KEYS:
                state = result.frame_heuristics.get(
                    key,
                    "indeterminate" if result.status == STATUS_INDETERMINATE else "fail",
                )
                row[key] = state
                if state == "pass":
                    heuristic_pass_counts[key] += 1
                    heuristic_evaluable_counts[key] += 1
                elif state != "indeterminate":
                    heuristic_evaluable_counts[key] += 1

            writer.writerow(row)
            status_counts[result.status] += 1
            confidences.append(result.confidence)

    _print_summary(
        total_frames=total_frames,
        status_counts=status_counts,
        confidences=confidences,
        heuristic_pass_counts=heuristic_pass_counts,
        heuristic_evaluable_counts=heuristic_evaluable_counts,
    )
    print(f"CSV written to: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 1 detector over a video and log frame-by-frame heuristic outcomes to CSV."
        )
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="validation_output.csv",
        help="Optional output CSV path (default: validation_output.csv)",
    )
    args = parser.parse_args()

    run_validation(video_path=args.video_path, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
