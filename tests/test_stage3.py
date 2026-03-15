from __future__ import annotations

from datetime import datetime
import json
import tempfile
import types
import unittest
from unittest.mock import Mock

import numpy as np

from detector import HeuristicResult
from stage3 import NtfyNotifier, UncertainFrameQueue, handle_stage3
from vlm import Stage2Result


def make_stage1_result() -> HeuristicResult:
    return HeuristicResult(
        flag=True,
        fired_heuristics=["h1_spine_vertical", "h3_head_bobbing"],
        confidence=0.8,
        active_count=2,
        heuristic_scores={
            "h1_spine_vertical": 0.91,
            "h2_wrist_oscillation": 0.10,
            "h3_head_bobbing": 0.78,
            "h5_head_tilt_back": 0.22,
        },
        frame_index=37,
        timestamp_sec=12.34,
    )


def make_stage2_result(verdict: str, confidence: float = 0.5) -> Stage2Result:
    return Stage2Result(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        raw_response=verdict,
        frame_index=37,
        timestamp_sec=12.34,
        provider_used="nebius",
    )


class Stage3Tests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((24, 24, 3), dtype=np.uint8)

    def test_handle_stage3_yes_routes_to_notifier_without_confidence_gate(self):
        notifier = Mock()
        notifier.send.return_value = True
        queue = Mock()

        stage2_result = make_stage2_result("YES", confidence=0.01)
        action = handle_stage3(
            frame=self.frame,
            stage1_result=make_stage1_result(),
            stage2_result=stage2_result,
            notifier=notifier,
            queue=queue,
        )

        self.assertEqual(action, "ALERT_SENT")
        notifier.send.assert_called_once()
        queue.save.assert_not_called()

    def test_handle_stage3_unsure_routes_to_queue(self):
        notifier = Mock()
        queue = Mock()

        stage2_result = make_stage2_result("UNSURE", confidence=0.5)
        action = handle_stage3(
            frame=self.frame,
            stage1_result=make_stage1_result(),
            stage2_result=stage2_result,
            notifier=notifier,
            queue=queue,
        )

        self.assertEqual(action, "QUEUED_UNSURE")
        notifier.send.assert_not_called()
        queue.save.assert_called_once()

    def test_handle_stage3_no_is_noop(self):
        notifier = Mock()
        queue = Mock()

        action = handle_stage3(
            frame=self.frame,
            stage1_result=make_stage1_result(),
            stage2_result=make_stage2_result("NO", confidence=0.99),
            notifier=notifier,
            queue=queue,
        )

        self.assertEqual(action, "NOOP_NO")
        notifier.send.assert_not_called()
        queue.save.assert_not_called()

    def test_uncertain_queue_saves_expected_name_and_sidecar_payload(self):
        fixed_now = datetime(2026, 3, 15, 14, 5, 6)
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = UncertainFrameQueue(queue_dir=tmp_dir, now_fn=lambda: fixed_now)
            stage1_result = make_stage1_result()
            stage2_result = make_stage2_result("UNSURE", confidence=0.5)

            image_path, json_path = queue.save(self.frame, stage1_result, stage2_result)

            self.assertEqual(
                image_path.name,
                "20260315_140506_t12.34_conf0.50.jpg",
            )
            self.assertEqual(
                json_path.name,
                "20260315_140506_t12.34_conf0.50.json",
            )
            self.assertTrue(image_path.exists())
            self.assertTrue(json_path.exists())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["frame_index"], 37)
            self.assertAlmostEqual(payload["timestamp_sec"], 12.34, places=6)
            self.assertEqual(payload["stage2_verdict"], "UNSURE")
            self.assertAlmostEqual(payload["stage2_confidence"], 0.5, places=6)
            self.assertEqual(payload["provider_used"], "nebius")
            self.assertEqual(payload["raw_response"], "UNSURE")
            self.assertEqual(payload["stage1_active_count"], 2)
            self.assertEqual(
                payload["stage1_fired_heuristics"],
                ["h1_spine_vertical", "h3_head_bobbing"],
            )
            self.assertEqual(payload["stage1_scores"], stage1_result.heuristic_scores)
            self.assertNotIn("heuristic_scores", payload)
            self.assertIn("saved_at", payload)

    def test_notifier_posts_text_payload(self):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

        def requester(url, data, headers, timeout):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            captured["timeout"] = timeout
            return FakeResponse()

        config = types.SimpleNamespace(
            ntfy_topic="pool-watch-alerts",
            vlm_timeout_seconds=9,
        )
        notifier = NtfyNotifier(config_module=config, requester=requester)
        sent = notifier.send(make_stage1_result(), make_stage2_result("YES", confidence=0.01))

        self.assertTrue(sent)
        self.assertEqual(captured["url"], "https://ntfy.sh/pool-watch-alerts")
        self.assertEqual(captured["timeout"], 9)
        self.assertIn("Title", captured["headers"])
        text = captured["data"].decode("utf-8")
        self.assertIn("stage2_verdict=YES", text)
        self.assertIn("stage2_confidence=0.01", text)
        self.assertIn("stage1_fired_heuristics=h1_spine_vertical,h3_head_bobbing", text)


if __name__ == "__main__":
    unittest.main()
