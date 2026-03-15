from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Literal, Tuple

import cv2
import numpy as np
import requests

from detector import HeuristicResult
from vlm import Stage2Result

Stage3Action = Literal["ALERT_SENT", "ALERT_FAILED", "QUEUED_UNSURE", "QUEUE_FAILED", "NOOP_NO"]


class NtfyNotifier:
    def __init__(self, config_module, requester=None):
        self.config = config_module
        self._requester = requester or requests.post

    def send(self, stage1_result: HeuristicResult, stage2_result: Stage2Result) -> bool:
        topic = str(getattr(self.config, "ntfy_topic", "") or "").strip()
        if not topic:
            print("[Stage3] ntfy_topic is empty; skipping alert send.")
            return False

        url = f"https://ntfy.sh/{topic}"
        message = (
            "POOL DISTRESS ALERT\n"
            f"timestamp={stage2_result.timestamp_sec:.2f}s\n"
            f"frame_index={stage2_result.frame_index}\n"
            f"provider={stage2_result.provider_used}\n"
            f"stage2_verdict={stage2_result.verdict}\n"
            f"stage2_confidence={stage2_result.confidence:.2f}\n"
            f"stage1_active_count={stage1_result.active_count}\n"
            f"stage1_fired_heuristics={','.join(stage1_result.fired_heuristics)}"
        )
        timeout_seconds = int(getattr(self.config, "vlm_timeout_seconds", 20))
        headers = {
            "Title": "Pool Watch Alert",
            "Tags": "warning,rotating_light",
            "Priority": "urgent",
        }

        try:
            response = self._requester(
                url,
                data=message.encode("utf-8"),
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            print(f"[Stage3] ntfy request failed: {type(exc).__name__}: {exc}")
            return False


class UncertainFrameQueue:
    def __init__(
        self,
        queue_dir: str | Path = "uncertain_frames",
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self._now_fn = now_fn or (lambda: datetime.now())

    def save(
        self,
        frame: np.ndarray,
        stage1_result: HeuristicResult,
        stage2_result: Stage2Result,
    ) -> Tuple[Path, Path]:
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        now = self._now_fn()
        prefix = now.strftime("%Y%m%d_%H%M%S")
        stem = (
            f"{prefix}_t{stage2_result.timestamp_sec:.2f}"
            f"_conf{stage2_result.confidence:.2f}"
        )
        image_path = self.queue_dir / f"{stem}.jpg"
        json_path = self.queue_dir / f"{stem}.json"

        if not cv2.imwrite(str(image_path), frame):
            raise ValueError(f"Failed to save uncertain frame: {image_path}")

        payload = {
            "frame_index": int(stage2_result.frame_index),
            "timestamp_sec": float(stage2_result.timestamp_sec),
            "saved_at": now.astimezone(timezone.utc).isoformat(),
            "stage2_verdict": stage2_result.verdict,
            "stage2_confidence": float(stage2_result.confidence),
            "provider_used": stage2_result.provider_used,
            "raw_response": stage2_result.raw_response,
            "stage1_active_count": int(stage1_result.active_count),
            "stage1_fired_heuristics": list(stage1_result.fired_heuristics),
            "stage1_scores": dict(stage1_result.heuristic_scores),
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return image_path, json_path


def handle_stage3(
    frame: np.ndarray,
    stage1_result: HeuristicResult,
    stage2_result: Stage2Result,
    notifier: NtfyNotifier,
    queue: UncertainFrameQueue,
) -> Stage3Action:
    if stage2_result.verdict == "YES":
        sent = notifier.send(stage1_result=stage1_result, stage2_result=stage2_result)
        return "ALERT_SENT" if sent else "ALERT_FAILED"

    if stage2_result.verdict == "UNSURE":
        try:
            queue.save(frame=frame, stage1_result=stage1_result, stage2_result=stage2_result)
            return "QUEUED_UNSURE"
        except Exception as exc:
            print(f"[Stage3] Failed to queue uncertain frame: {type(exc).__name__}: {exc}")
            return "QUEUE_FAILED"

    return "NOOP_NO"
