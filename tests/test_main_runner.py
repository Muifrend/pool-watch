import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from detector import HeuristicResult
import main as main_module
from main import VideoRunner
from vlm import Stage2Result


class StubDetector:
    def __init__(self):
        self.frames = []

    def process_frame(self, frame):
        self.frames.append(frame)
        return {"flag": False}


class VideoRunnerTests(unittest.TestCase):
    @patch("main.cv2.VideoCapture")
    def test_loop_seek_calls_set_pos_frames_explicitly(self, mock_capture_cls):
        frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_b = np.ones((8, 8, 3), dtype=np.uint8)
        frame_c = np.full((8, 8, 3), 2, dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame_a), (True, frame_b), (True, frame_c)]

        def get_side_effect(prop):
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            return 0.0

        mock_cap.get.side_effect = get_side_effect
        mock_capture_cls.return_value = mock_cap

        detector = StubDetector()
        runner = VideoRunner(video_path="pool_video.mp4", loop_start_sec=0.0, loop_end_sec=0.2)

        outputs = list(runner.run(detector=detector, max_frames=3))

        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(detector.frames), 3)
        self.assertGreaterEqual(mock_cap.set.call_count, 2)
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 0)

    @patch("main.cv2.VideoCapture")
    def test_frame_skip_is_applied_by_runner(self, mock_capture_cls):
        frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(6)]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame) for frame in frames]

        def get_side_effect(prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        mock_cap.get.side_effect = get_side_effect
        mock_capture_cls.return_value = mock_cap

        detector = StubDetector()
        runner = VideoRunner(video_path="pool_video.mp4", frame_skip=1)

        outputs = list(runner.run(detector=detector, max_frames=3))

        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(detector.frames), 3)
        self.assertTrue(np.array_equal(detector.frames[0], frames[0]))
        self.assertTrue(np.array_equal(detector.frames[1], frames[2]))
        self.assertTrue(np.array_equal(detector.frames[2], frames[4]))
        self.assertEqual(mock_cap.release.call_count, 1)

    @patch("main.cv2.VideoCapture")
    def test_runner_populates_frame_index_and_timestamp(self, mock_capture_cls):
        frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_b = np.ones((8, 8, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame_a), (True, frame_b)]

        def get_side_effect(prop):
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            return 0.0

        mock_cap.get.side_effect = get_side_effect
        mock_capture_cls.return_value = mock_cap

        class ResultDetector:
            def process_frame(self, _frame):
                return HeuristicResult(
                    flag=False,
                    fired_heuristics=[],
                    confidence=0.0,
                    active_count=0,
                    heuristic_scores={},
                )

        outputs = list(
            VideoRunner(video_path="pool_video.mp4").run(
                detector=ResultDetector(),
                max_frames=2,
            )
        )
        result0 = outputs[0][1]
        result1 = outputs[1][1]

        self.assertEqual(result0.frame_index, 0)
        self.assertEqual(result0.timestamp_sec, 0.0)
        self.assertEqual(result1.frame_index, 1)
        self.assertAlmostEqual(result1.timestamp_sec, 0.1, places=6)

    @patch("main.handle_stage3")
    @patch("main.UncertainFrameQueue")
    @patch("main.NtfyNotifier")
    @patch("main.Stage2Confirmer")
    @patch("main.run_stage1")
    def test_main_routes_only_flagged_frames_into_stage2_stage3(
        self,
        mock_run_stage1,
        mock_confirmer_cls,
        mock_notifier_cls,
        mock_queue_cls,
        mock_handle_stage3,
    ):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        flagged = HeuristicResult(
            flag=True,
            fired_heuristics=["h1_spine_vertical"],
            confidence=0.9,
            active_count=1,
            heuristic_scores={"h1_spine_vertical": 1.0},
        )
        not_flagged = HeuristicResult(
            flag=False,
            fired_heuristics=[],
            confidence=0.0,
            active_count=0,
            heuristic_scores={"h1_spine_vertical": 0.0},
        )
        mock_run_stage1.return_value = iter([(frame, flagged), (frame, not_flagged)])

        stage2_result = Stage2Result(
            verdict="YES",
            confidence=0.01,
            raw_response="YES",
            frame_index=4,
            timestamp_sec=1.2,
            provider_used="nebius",
        )
        confirmer = mock_confirmer_cls.return_value
        confirmer.confirm.return_value = stage2_result
        mock_handle_stage3.return_value = "ALERT_SENT"

        main_module.main()

        mock_run_stage1.assert_called_once_with(max_frames=300)
        confirmer.confirm.assert_called_once_with(frame, flagged)
        mock_handle_stage3.assert_called_once_with(
            frame=frame,
            stage1_result=flagged,
            stage2_result=stage2_result,
            notifier=mock_notifier_cls.return_value,
            queue=mock_queue_cls.return_value,
        )


if __name__ == "__main__":
    unittest.main()
