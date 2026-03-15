from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

import config

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover - keep tests runnable with injected fakes
    mp = None
    mp_python = None
    mp_vision = None

HEURISTIC_KEYS = (
    "h1_spine_vertical",
    "h2_wrist_oscillation",
    "h3_head_bobbing",
    "h5_head_tilt_back",
)

STATUS_DETERMINED = "DETERMINED"
STATUS_INDETERMINATE = "INDETERMINATE"

NOSE_IDX = 0
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

LANDMARK_INDEX = {
    "NOSE": NOSE_IDX,
    "LEFT_EAR": LEFT_EAR_IDX,
    "RIGHT_EAR": RIGHT_EAR_IDX,
    "LEFT_SHOULDER": LEFT_SHOULDER_IDX,
    "RIGHT_SHOULDER": RIGHT_SHOULDER_IDX,
    "LEFT_WRIST": LEFT_WRIST_IDX,
    "RIGHT_WRIST": RIGHT_WRIST_IDX,
    "LEFT_HIP": LEFT_HIP_IDX,
    "RIGHT_HIP": RIGHT_HIP_IDX,
}

LandmarkPoint = Tuple[float, float, float]


@dataclass(frozen=True)
class HeuristicResult:
    flag: bool
    fired_heuristics: List[str]
    confidence: float
    active_count: int
    heuristic_scores: Dict[str, float]
    frame_heuristics: Dict[str, str] = field(default_factory=dict)
    status: str = STATUS_DETERMINED


class Stage1Detector:
    """Stage 1 detector that evaluates distress heuristics on one frame at a time."""

    def __init__(
        self,
        visibility_threshold: float = 0.6,
        heuristics_window_seconds: float = 3.0,
        min_heuristics: int = 3,
        fps: float = 30.0,
        activation_ratio_threshold: float = 0.6,
        pose: Optional[object] = None,
        pose_model_path: Optional[str] = None,
        persistence_window_frames: Optional[Dict[str, int]] = None,
        persistence_min_passes: Optional[Dict[str, int]] = None,
    ) -> None:
        self.visibility_threshold = float(visibility_threshold)
        self.min_heuristics = int(min_heuristics)
        self.activation_ratio_threshold = float(activation_ratio_threshold)
        self._heuristics_window_seconds = float(heuristics_window_seconds)

        effective_fps = fps if fps and fps > 0 else 30.0
        self._effective_fps = effective_fps
        self._window_size = max(1, int(round(heuristics_window_seconds * effective_fps)))
        self._h2_recent_window_size = max(
            6,
            int(round(min(1.0, self._heuristics_window_seconds) * self._effective_fps)),
        )

        self._heuristics_window: Deque[Dict[str, bool]] = deque(maxlen=self._window_size)
        self._wrist_rel_history: Deque[Optional[float]] = deque(maxlen=self._window_size)
        self._nose_offset_history: Deque[Optional[float]] = deque(maxlen=self._window_size)

        default_window_frames = {
            "h1_spine_vertical": config.h1_persistence_window_frames,
            "h2_wrist_oscillation": config.h2_persistence_window_frames,
            "h3_head_bobbing": config.h3_persistence_window_frames,
            "h5_head_tilt_back": config.h5_persistence_window_frames,
        }
        default_min_passes = {
            "h1_spine_vertical": config.h1_persistence_min_passes,
            "h2_wrist_oscillation": config.h2_persistence_min_passes,
            "h3_head_bobbing": config.h3_persistence_min_passes,
            "h5_head_tilt_back": config.h5_persistence_min_passes,
        }
        self._persistence_window_frames = (
            dict(persistence_window_frames)
            if persistence_window_frames is not None
            else default_window_frames
        )
        self._persistence_min_passes = (
            dict(persistence_min_passes)
            if persistence_min_passes is not None
            else default_min_passes
        )
        self._validate_persistence_config()
        self._persistence_history = {
            key: deque(maxlen=self._persistence_window_frames[key]) for key in HEURISTIC_KEYS
        }

        model_path = pose_model_path or config.pose_landmarker_model_path
        self._pose = pose if pose is not None else self._build_default_pose(model_path)

    def process_frame(self, frame: np.ndarray) -> HeuristicResult:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must have shape (H, W, >=3)")

        rgb_frame = np.ascontiguousarray(frame[..., :3][..., ::-1])
        pose_output = self._detect_pose(rgb_frame)

        people_landmarks = list(getattr(pose_output, "pose_landmarks", []) or [])
        landmarks = people_landmarks[0] if people_landmarks else None

        raw_frame_state = self._compute_heuristics(landmarks)
        smoothed_frame_state = self._apply_persistence(raw_frame_state)
        self._heuristics_window.append(smoothed_frame_state)

        scores = self._compute_activation_scores()
        fired_heuristics = [
            key for key in HEURISTIC_KEYS if scores[key] >= self.activation_ratio_threshold
        ]
        active_count = len(fired_heuristics)
        flag = active_count >= self.min_heuristics
        confidence = float(sum(scores.values()) / len(HEURISTIC_KEYS))

        status = STATUS_DETERMINED if landmarks is not None else STATUS_INDETERMINATE
        if status == STATUS_INDETERMINATE:
            frame_heuristics = {key: "indeterminate" for key in HEURISTIC_KEYS}
        else:
            frame_heuristics = {
                key: ("pass" if smoothed_frame_state[key] else "fail") for key in HEURISTIC_KEYS
            }

        return HeuristicResult(
            flag=flag,
            fired_heuristics=fired_heuristics,
            confidence=confidence,
            active_count=active_count,
            heuristic_scores=scores,
            frame_heuristics=frame_heuristics,
            status=status,
        )

    def _validate_persistence_config(self) -> None:
        missing_window = [k for k in HEURISTIC_KEYS if k not in self._persistence_window_frames]
        missing_min = [k for k in HEURISTIC_KEYS if k not in self._persistence_min_passes]
        if missing_window or missing_min:
            raise ValueError(
                "Persistence config missing heuristics: "
                f"window_missing={missing_window}, min_missing={missing_min}"
            )

        for key in HEURISTIC_KEYS:
            window_n = int(self._persistence_window_frames[key])
            min_passes = int(self._persistence_min_passes[key])
            if window_n <= 0:
                raise ValueError(f"Persistence window must be > 0 for {key}")
            if min_passes <= 0:
                raise ValueError(f"Persistence min_passes must be > 0 for {key}")
            if min_passes > window_n:
                raise ValueError(
                    f"Persistence min_passes ({min_passes}) cannot exceed window ({window_n}) for {key}"
                )
            self._persistence_window_frames[key] = window_n
            self._persistence_min_passes[key] = min_passes

    def _apply_persistence(self, frame_state: Dict[str, bool]) -> Dict[str, bool]:
        smoothed_state: Dict[str, bool] = {}
        for key in HEURISTIC_KEYS:
            history = self._persistence_history[key]
            history.append(bool(frame_state.get(key, False)))
            pass_count = sum(1 for value in history if value)
            smoothed_state[key] = pass_count >= self._persistence_min_passes[key]
        return smoothed_state

    def _build_default_pose(self, model_path: str) -> object:
        if mp_python is None or mp_vision is None:
            raise RuntimeError(
                "MediaPipe Tasks API is unavailable in this environment. "
                "Inject a PoseLandmarker-like object via `pose`, or install mediapipe>=0.10 "
                "with tasks support."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    def _detect_pose(self, rgb_frame: np.ndarray) -> object:
        if hasattr(self._pose, "detect"):
            if mp is None:
                raise RuntimeError("mediapipe package is required to create mp.Image inputs")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            return self._pose.detect(mp_image)

        # Backward-compatible adapter path for tests or injected legacy fakes.
        if hasattr(self._pose, "process"):
            return self._pose.process(rgb_frame)

        raise TypeError("Injected pose object must expose detect(mp_image) or process(rgb_frame)")

    def _compute_heuristics(self, landmarks: Optional[Iterable[object]]) -> Dict[str, bool]:
        if landmarks is None:
            self._wrist_rel_history.append(None)
            self._nose_offset_history.append(None)
            return {key: False for key in HEURISTIC_KEYS}

        landmark_list = list(landmarks)
        get = lambda name: self._point_from_landmarks(landmark_list, name)

        left_shoulder = get("LEFT_SHOULDER")
        right_shoulder = get("RIGHT_SHOULDER")
        left_hip = get("LEFT_HIP")
        right_hip = get("RIGHT_HIP")
        left_wrist = get("LEFT_WRIST")
        right_wrist = get("RIGHT_WRIST")
        nose = get("NOSE")
        left_ear = get("LEFT_EAR")
        right_ear = get("RIGHT_EAR")

        shoulders_mid = self._midpoint(left_shoulder, right_shoulder)
        hips_mid = self._midpoint(left_hip, right_hip)
        wrists_mid = self._midpoint(left_wrist, right_wrist)

        wrist_rel: Optional[float] = None
        if shoulders_mid is not None:
            if wrists_mid is not None:
                wrist_rel = wrists_mid[1] - shoulders_mid[1]
            elif left_wrist is not None:
                wrist_rel = left_wrist[1] - shoulders_mid[1]
            elif right_wrist is not None:
                wrist_rel = right_wrist[1] - shoulders_mid[1]
        self._wrist_rel_history.append(wrist_rel)

        nose_offset: Optional[float] = None
        if shoulders_mid is not None and nose is not None:
            nose_offset = nose[1] - shoulders_mid[1]
        self._nose_offset_history.append(nose_offset)

        return {
            "h1_spine_vertical": self._is_spine_vertical(shoulders_mid, hips_mid),
            "h2_wrist_oscillation": self._is_wrist_oscillating(wrist_rel),
            "h3_head_bobbing": self._is_head_bobbing(),
            "h5_head_tilt_back": self._is_head_tilt_back(
                nose=nose,
                left_ear=left_ear,
                right_ear=right_ear,
                left_shoulder=left_shoulder,
                right_shoulder=right_shoulder,
                shoulders_mid=shoulders_mid,
            ),
        }

    def _point_from_landmarks(
        self, landmarks: List[object], landmark_name: str
    ) -> Optional[LandmarkPoint]:
        index = LANDMARK_INDEX.get(landmark_name)
        if index is None or index >= len(landmarks):
            return None

        lm = landmarks[index]
        visibility = getattr(lm, "visibility", 0.0)
        if visibility < self.visibility_threshold:
            return None
        return (float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0)))

    @staticmethod
    def _midpoint(p1: Optional[LandmarkPoint], p2: Optional[LandmarkPoint]) -> Optional[LandmarkPoint]:
        if p1 is None or p2 is None:
            return None
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)

    @staticmethod
    def _is_spine_vertical(
        shoulders_mid: Optional[LandmarkPoint], hips_mid: Optional[LandmarkPoint]
    ) -> bool:
        if shoulders_mid is None or hips_mid is None:
            return False

        vx = hips_mid[0] - shoulders_mid[0]
        vy = hips_mid[1] - shoulders_mid[1]
        angle = abs(math.degrees(math.atan2(vy, vx)))
        if angle > 90.0:
            angle = 180.0 - angle
        return 70.0 <= angle <= 90.0

    def _is_wrist_oscillating(self, wrist_rel: Optional[float]) -> bool:
        if wrist_rel is None:
            return False

        recent_values = list(self._wrist_rel_history)[-self._h2_recent_window_size :]
        values = [value for value in recent_values if value is not None]
        if len(values) < 6:
            return False

        amplitude = max(values) - min(values)
        if amplitude < 0.015:
            return False

        signs: List[int] = []
        for value in values:
            if abs(value) < 0.005:
                continue
            signs.append(1 if value > 0 else -1)
        if len(signs) < 3:
            return False

        sign_changes = sum(1 for prev, nxt in zip(signs, signs[1:]) if prev != nxt)
        at_shoulder_level = abs(wrist_rel) <= 0.14
        return at_shoulder_level and sign_changes >= 2

    def _is_head_bobbing(self) -> bool:
        values = [value for value in self._nose_offset_history if value is not None]
        if len(values) < 6:
            return False
        return (max(values) - min(values)) >= 0.04

    @staticmethod
    def _is_head_tilt_back(
        nose: Optional[LandmarkPoint],
        left_ear: Optional[LandmarkPoint],
        right_ear: Optional[LandmarkPoint],
        left_shoulder: Optional[LandmarkPoint],
        right_shoulder: Optional[LandmarkPoint],
        shoulders_mid: Optional[LandmarkPoint],
    ) -> bool:
        if (
            nose is None
        ):
            return False

        ear_candidates = [ear[1] for ear in (left_ear, right_ear) if ear is not None]
        if not ear_candidates:
            return False
        ear_reference_y = sum(ear_candidates) / len(ear_candidates)

        shoulder_candidates = [s[1] for s in (left_shoulder, right_shoulder) if s is not None]
        if shoulders_mid is not None:
            shoulder_reference_y = shoulders_mid[1]
        elif shoulder_candidates:
            shoulder_reference_y = sum(shoulder_candidates) / len(shoulder_candidates)
        else:
            return False

        if left_shoulder is not None and right_shoulder is not None:
            shoulders_level = abs(left_shoulder[1] - right_shoulder[1]) <= 0.12
        else:
            shoulders_level = True

        nose_near_or_above_ears = nose[1] <= ear_reference_y + 0.03
        nose_above_shoulders = nose[1] + 0.03 < shoulder_reference_y
        return shoulders_level and nose_near_or_above_ears and nose_above_shoulders

    def _compute_activation_scores(self) -> Dict[str, float]:
        if not self._heuristics_window:
            return {key: 0.0 for key in HEURISTIC_KEYS}

        window_len = len(self._heuristics_window)
        scores: Dict[str, float] = {}
        for key in HEURISTIC_KEYS:
            active_count = sum(1 for frame_state in self._heuristics_window if frame_state[key])
            scores[key] = active_count / window_len
        return scores
