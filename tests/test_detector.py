import types
import unittest

import numpy as np

from detector import HEURISTIC_KEYS, HeuristicResult, LANDMARK_INDEX, Stage1Detector


class FakePose:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._index = 0

    def detect(self, _mp_image):
        if self._index < len(self._outputs):
            output = self._outputs[self._index]
            self._index += 1
            return output
        if not self._outputs:
            return types.SimpleNamespace(pose_landmarks=[])
        return self._outputs[-1]


def make_pose_output(landmarks):
    if landmarks is None:
        return types.SimpleNamespace(pose_landmarks=[])
    return types.SimpleNamespace(pose_landmarks=[landmarks])


def _landmark(x=0.0, y=0.0, z=0.0, visibility=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def build_landmarks(**overrides):
    values = [_landmark() for _ in range(33)]
    for name, payload in overrides.items():
        x, y, visibility = payload
        index = LANDMARK_INDEX[name]
        values[index] = _landmark(x=x, y=y, visibility=visibility)
    return values


def base_landmarks(
    *,
    visibility=1.0,
    hip_left=(0.45, 0.8),
    hip_right=(0.55, 0.8),
    wrist_y_left=0.5,
    wrist_y_right=0.5,
    nose_y=0.3,
    left_ear_y=0.34,
    right_ear_y=0.34,
):
    return build_landmarks(
        LEFT_SHOULDER=(0.4, 0.5, visibility),
        RIGHT_SHOULDER=(0.6, 0.5, visibility),
        LEFT_HIP=(hip_left[0], hip_left[1], visibility),
        RIGHT_HIP=(hip_right[0], hip_right[1], visibility),
        LEFT_WRIST=(0.4, wrist_y_left, visibility),
        RIGHT_WRIST=(0.6, wrist_y_right, visibility),
        NOSE=(0.5, nose_y, visibility),
        LEFT_EAR=(0.45, left_ear_y, visibility),
        RIGHT_EAR=(0.55, right_ear_y, visibility),
    )


class Stage1DetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)

    def _detector(self, outputs, **kwargs):
        defaults = {
            "visibility_threshold": 0.6,
            "heuristics_window_seconds": 1.0,
            "min_heuristics": 3,
            "fps": 8,
            "activation_ratio_threshold": 0.6,
            "pose": FakePose(outputs),
        }
        defaults.update(kwargs)
        return Stage1Detector(**defaults)

    def test_process_frame_returns_expected_contract(self):
        detector = self._detector([make_pose_output(None)])
        result = detector.process_frame(self.frame)

        self.assertIsInstance(result, HeuristicResult)
        self.assertEqual(set(result.heuristic_scores.keys()), set(HEURISTIC_KEYS))
        self.assertEqual(result.fired_heuristics, [])
        self.assertFalse(result.flag)
        self.assertEqual(result.active_count, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_visibility_threshold_boundary(self):
        visible = base_landmarks(visibility=0.6)
        below = base_landmarks(visibility=0.59)

        detector_visible = self._detector(
            [make_pose_output(visible)],
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
        )
        detector_below = self._detector(
            [make_pose_output(below)],
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
        )

        visible_result = detector_visible.process_frame(self.frame)
        below_result = detector_below.process_frame(self.frame)

        self.assertEqual(visible_result.heuristic_scores["h1_spine_vertical"], 1.0)
        self.assertEqual(below_result.heuristic_scores["h1_spine_vertical"], 0.0)

    def test_rolling_window_accumulates_and_decays(self):
        outputs = [make_pose_output(None) for _ in range(10)]
        detector = self._detector(outputs, fps=5, heuristics_window_seconds=1.0)

        state_true = {
            "h1_spine_vertical": True,
            "h2_wrist_oscillation": False,
            "h3_head_bobbing": False,
            "h5_head_tilt_back": False,
        }
        state_false = {key: False for key in HEURISTIC_KEYS}
        states = [state_true] * 5 + [state_false] * 5
        iterator = iter(states)
        detector._compute_heuristics = lambda _landmarks: next(iterator)

        for _ in range(5):
            mid_result = detector.process_frame(self.frame)
        self.assertEqual(mid_result.heuristic_scores["h1_spine_vertical"], 1.0)

        for _ in range(5):
            end_result = detector.process_frame(self.frame)
        self.assertEqual(end_result.heuristic_scores["h1_spine_vertical"], 0.0)

    def test_flag_threshold_with_2_3_and_4_active_heuristics(self):
        outputs = [make_pose_output(None) for _ in range(3)]
        detector = self._detector(
            outputs,
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=3,
        )
        states = iter(
            [
                {
                    "h1_spine_vertical": True,
                    "h2_wrist_oscillation": True,
                    "h3_head_bobbing": False,
                    "h5_head_tilt_back": False,
                },
                {
                    "h1_spine_vertical": True,
                    "h2_wrist_oscillation": True,
                    "h3_head_bobbing": True,
                    "h5_head_tilt_back": False,
                },
                {
                    "h1_spine_vertical": True,
                    "h2_wrist_oscillation": True,
                    "h3_head_bobbing": True,
                    "h5_head_tilt_back": True,
                },
            ]
        )
        detector._compute_heuristics = lambda _landmarks: next(states)

        result_2 = detector.process_frame(self.frame)
        result_3 = detector.process_frame(self.frame)
        result_4 = detector.process_frame(self.frame)

        self.assertFalse(result_2.flag)
        self.assertEqual(result_2.active_count, 2)
        self.assertTrue(result_3.flag)
        self.assertEqual(result_3.active_count, 3)
        self.assertTrue(result_4.flag)
        self.assertEqual(result_4.active_count, 4)

    def test_confidence_uses_window_ratio_average(self):
        outputs = [make_pose_output(None) for _ in range(4)]
        detector = self._detector(
            outputs,
            fps=1,
            heuristics_window_seconds=4.0,
            min_heuristics=3,
        )
        states = iter(
            [
                {
                    "h1_spine_vertical": True,
                    "h2_wrist_oscillation": False,
                    "h3_head_bobbing": False,
                    "h5_head_tilt_back": False,
                },
                {
                    "h1_spine_vertical": False,
                    "h2_wrist_oscillation": True,
                    "h3_head_bobbing": False,
                    "h5_head_tilt_back": False,
                },
                {
                    "h1_spine_vertical": False,
                    "h2_wrist_oscillation": False,
                    "h3_head_bobbing": True,
                    "h5_head_tilt_back": False,
                },
                {
                    "h1_spine_vertical": False,
                    "h2_wrist_oscillation": False,
                    "h3_head_bobbing": False,
                    "h5_head_tilt_back": True,
                },
            ]
        )
        detector._compute_heuristics = lambda _landmarks: next(states)

        for _ in range(4):
            result = detector.process_frame(self.frame)

        self.assertAlmostEqual(result.confidence, 0.25, places=6)
        for score in result.heuristic_scores.values():
            self.assertAlmostEqual(score, 0.25, places=6)

    def test_h1_spine_vertical_positive_and_negative(self):
        vertical = base_landmarks(hip_left=(0.45, 0.82), hip_right=(0.55, 0.82))
        non_vertical = base_landmarks(hip_left=(0.8, 0.54), hip_right=(0.9, 0.56))

        detector_pos = self._detector(
            [make_pose_output(vertical)],
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
        )
        detector_neg = self._detector(
            [make_pose_output(non_vertical)],
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
        )

        result_pos = detector_pos.process_frame(self.frame)
        result_neg = detector_neg.process_frame(self.frame)

        self.assertEqual(result_pos.heuristic_scores["h1_spine_vertical"], 1.0)
        self.assertEqual(result_neg.heuristic_scores["h1_spine_vertical"], 0.0)

    def test_h2_wrist_oscillation_positive_and_negative(self):
        positive_rel = [-0.03, 0.03, -0.03, 0.03, -0.03, 0.03, -0.03, 0.03]
        negative_rel = [0.01] * 8

        positive_outputs = [
            make_pose_output(
                base_landmarks(
                    hip_left=(0.8, 0.54),
                    hip_right=(0.9, 0.56),
                    wrist_y_left=0.5 + rel,
                    wrist_y_right=0.5 + rel,
                    nose_y=0.38,
                    left_ear_y=0.32,
                    right_ear_y=0.32,
                )
            )
            for rel in positive_rel
        ]
        negative_outputs = [
            make_pose_output(
                base_landmarks(
                    hip_left=(0.8, 0.54),
                    hip_right=(0.9, 0.56),
                    wrist_y_left=0.5 + rel,
                    wrist_y_right=0.5 + rel,
                    nose_y=0.38,
                    left_ear_y=0.32,
                    right_ear_y=0.32,
                )
            )
            for rel in negative_rel
        ]

        detector_pos = self._detector(
            positive_outputs,
            fps=8,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
            activation_ratio_threshold=0.3,
        )
        detector_neg = self._detector(
            negative_outputs,
            fps=8,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
            activation_ratio_threshold=0.3,
        )

        for _ in range(8):
            result_pos = detector_pos.process_frame(self.frame)
            result_neg = detector_neg.process_frame(self.frame)

        self.assertGreater(result_pos.heuristic_scores["h2_wrist_oscillation"], 0.0)
        self.assertEqual(result_neg.heuristic_scores["h2_wrist_oscillation"], 0.0)

    def test_h3_head_bobbing_positive_and_negative(self):
        positive_nose_values = [0.30, 0.42, 0.31, 0.41, 0.29, 0.43, 0.30, 0.42]
        negative_nose_values = [0.36] * 8

        positive_outputs = [
            make_pose_output(
                base_landmarks(
                    hip_left=(0.8, 0.54),
                    hip_right=(0.9, 0.56),
                    wrist_y_left=0.8,
                    wrist_y_right=0.8,
                    nose_y=nose_y,
                    left_ear_y=0.33,
                    right_ear_y=0.33,
                )
            )
            for nose_y in positive_nose_values
        ]
        negative_outputs = [
            make_pose_output(
                base_landmarks(
                    hip_left=(0.8, 0.54),
                    hip_right=(0.9, 0.56),
                    wrist_y_left=0.8,
                    wrist_y_right=0.8,
                    nose_y=nose_y,
                    left_ear_y=0.33,
                    right_ear_y=0.33,
                )
            )
            for nose_y in negative_nose_values
        ]

        detector_pos = self._detector(
            positive_outputs,
            fps=8,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
            activation_ratio_threshold=0.3,
        )
        detector_neg = self._detector(
            negative_outputs,
            fps=8,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
            activation_ratio_threshold=0.3,
        )

        for _ in range(8):
            result_pos = detector_pos.process_frame(self.frame)
            result_neg = detector_neg.process_frame(self.frame)

        self.assertGreater(result_pos.heuristic_scores["h3_head_bobbing"], 0.0)
        self.assertEqual(result_neg.heuristic_scores["h3_head_bobbing"], 0.0)

    def test_h5_head_tilt_back_positive_and_negative(self):
        positive = base_landmarks(nose_y=0.16, left_ear_y=0.24, right_ear_y=0.24)
        negative = base_landmarks(nose_y=0.28, left_ear_y=0.24, right_ear_y=0.24)

        detector_pos = self._detector(
            [make_pose_output(positive)],
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
        )
        detector_neg = self._detector(
            [make_pose_output(negative)],
            fps=1,
            heuristics_window_seconds=1.0,
            min_heuristics=1,
        )

        result_pos = detector_pos.process_frame(self.frame)
        result_neg = detector_neg.process_frame(self.frame)

        self.assertEqual(result_pos.heuristic_scores["h5_head_tilt_back"], 1.0)
        self.assertEqual(result_neg.heuristic_scores["h5_head_tilt_back"], 0.0)


if __name__ == "__main__":
    unittest.main()
