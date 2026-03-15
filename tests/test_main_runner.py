import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from main import VideoRunner


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


if __name__ == "__main__":
    unittest.main()
