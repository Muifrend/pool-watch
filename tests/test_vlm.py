import types
import unittest
from unittest.mock import Mock

import numpy as np
import requests

from detector import HeuristicResult
from vlm import PROMPT, Stage2Confirmer, Stage2Result


class FakeResponse:
    def __init__(self, payload, should_raise=False):
        self._payload = payload
        self._should_raise = should_raise

    def raise_for_status(self):
        if self._should_raise:
            raise requests.HTTPError("bad status")

    def json(self):
        return self._payload


def make_config(provider="nebius"):
    return types.SimpleNamespace(
        vlm_provider=provider,
        nebius_base_url="https://nebius.example/v1/chat/completions",
        nebius_api_key="nebius-key",
        nebius_model="nebius-model",
        openrouter_base_url="https://openrouter.example/v1/chat/completions",
        openrouter_api_key="openrouter-key",
        openrouter_model="openrouter-model",
        vlm_frame_size=64,
        vlm_timeout_seconds=9,
        VERDICT_CONFIDENCE_MAP={"YES": 1.0, "NO": 0.0, "UNSURE": 0.5},
    )


def make_stage1_result(frame_index=12, timestamp_sec=1.23):
    return HeuristicResult(
        flag=True,
        fired_heuristics=["h1_spine_vertical"],
        confidence=0.8,
        active_count=1,
        heuristic_scores={"h1_spine_vertical": 1.0},
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


class Stage2ConfirmerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((24, 24, 3), dtype=np.uint8)

    def test_prompt_contract_has_no_bias_language(self):
        captured = {}

        def requester(url, headers, json, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse({"choices": [{"message": {"content": "YES"}}]})

        confirmer = Stage2Confirmer(make_config(), requester=requester, clock_fn=lambda: 100.0)
        _ = confirmer.confirm(self.frame, make_stage1_result())

        prompt = captured["json"]["messages"][0]["content"][0]["text"]
        self.assertEqual(prompt, PROMPT)

        lower_prompt = prompt.lower()
        for forbidden in ["spine", "wrist", "stroke", "bobbing", "tilted"]:
            self.assertNotIn(forbidden, lower_prompt)

    def test_response_parsing_matrix(self):
        cases = [
            ("YES", "YES", "YES"),
            ("NO", "NO", "NO"),
            ("UNSURE", "UNSURE", "UNSURE"),
            ("yes", "YES", "yes"),
            ("garbage", "UNSURE", "__parse_failure__"),
            ("", "UNSURE", "__parse_failure__"),
        ]

        for model_text, expected_verdict, expected_raw in cases:
            with self.subTest(model_text=model_text):
                requester = lambda *_args, **_kwargs: FakeResponse(
                    {"choices": [{"message": {"content": model_text}}]}
                )
                confirmer = Stage2Confirmer(
                    make_config(), requester=requester, clock_fn=lambda: 100.0
                )
                result = confirmer.confirm(self.frame, make_stage1_result())

                self.assertEqual(result.verdict, expected_verdict)
                self.assertEqual(result.raw_response, expected_raw)

    def test_timeout_returns_unsure_without_raise(self):
        def timeout_requester(*_args, **_kwargs):
            raise requests.Timeout("timed out")

        confirmer = Stage2Confirmer(
            make_config(), requester=timeout_requester, clock_fn=lambda: 100.0
        )

        result = confirmer.confirm(self.frame, make_stage1_result())
        self.assertEqual(result.verdict, "UNSURE")
        self.assertEqual(result.confidence, 0.5)
        self.assertEqual(result.raw_response, "__timeout__")

    def test_exception_returns_unsure_without_raise(self):
        def broken_requester(*_args, **_kwargs):
            raise RuntimeError("boom")

        confirmer = Stage2Confirmer(
            make_config(), requester=broken_requester, clock_fn=lambda: 100.0
        )

        result = confirmer.confirm(self.frame, make_stage1_result())
        self.assertEqual(result.verdict, "UNSURE")
        self.assertEqual(result.confidence, 0.5)
        self.assertEqual(result.raw_response, "__exception__")

    def test_http_error_returns_unsure_without_raise(self):
        err = requests.HTTPError("unauthorized")
        err.response = types.SimpleNamespace(status_code=401)

        def http_error_requester(*_args, **_kwargs):
            raise err

        confirmer = Stage2Confirmer(
            make_config(), requester=http_error_requester, clock_fn=lambda: 100.0
        )

        result = confirmer.confirm(self.frame, make_stage1_result())
        self.assertEqual(result.verdict, "UNSURE")
        self.assertEqual(result.confidence, 0.5)
        self.assertEqual(result.raw_response, "__exception__")

    def test_provider_routing_uses_configured_url(self):
        for provider, expected_url in [
            ("nebius", "https://nebius.example/v1/chat/completions"),
            ("openrouter", "https://openrouter.example/v1/chat/completions"),
        ]:
            with self.subTest(provider=provider):
                seen = {"url": None}

                def requester(url, headers, json, timeout):
                    seen["url"] = url
                    return FakeResponse({"choices": [{"message": {"content": "NO"}}]})

                confirmer = Stage2Confirmer(
                    make_config(provider=provider),
                    requester=requester,
                    clock_fn=lambda: 100.0,
                )
                _ = confirmer.confirm(self.frame, make_stage1_result())
                self.assertEqual(seen["url"], expected_url)

    def test_output_contract_is_fully_populated(self):
        requester = lambda *_args, **_kwargs: FakeResponse(
            {"choices": [{"message": {"content": "UNSURE"}}]}
        )
        confirmer = Stage2Confirmer(make_config(), requester=requester, clock_fn=lambda: 100.0)

        result = confirmer.confirm(self.frame, make_stage1_result(frame_index=44, timestamp_sec=2.2))
        self.assertIsInstance(result, Stage2Result)
        self.assertIn(result.verdict, {"YES", "NO", "UNSURE"})
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.raw_response, str)
        self.assertEqual(result.frame_index, 44)
        self.assertEqual(result.timestamp_sec, 2.2)
        self.assertIn(result.provider_used, {"nebius", "openrouter"})

    def test_debounce_uses_monotonic_even_if_video_timestamp_resets(self):
        requester = Mock(
            side_effect=[
                FakeResponse({"choices": [{"message": {"content": "YES"}}]}),
                FakeResponse({"choices": [{"message": {"content": "YES"}}]}),
            ]
        )
        clock_values = iter([100.0, 105.0])
        confirmer = Stage2Confirmer(
            make_config(),
            requester=requester,
            clock_fn=lambda: next(clock_values),
        )

        result1 = confirmer.confirm(self.frame, make_stage1_result(frame_index=10, timestamp_sec=3.0))
        result2 = confirmer.confirm(self.frame, make_stage1_result(frame_index=11, timestamp_sec=0.0))

        self.assertEqual(result1.verdict, "YES")
        self.assertEqual(result2.verdict, "NO")
        self.assertEqual(result2.raw_response, "__debounced__")
        self.assertEqual(requester.call_count, 1)


if __name__ == "__main__":
    unittest.main()
