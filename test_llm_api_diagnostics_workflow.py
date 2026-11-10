import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import yaml


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github/workflows/llm-api-diagnostics.yml"


class LlmApiDiagnosticsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        step = next(
            step for step in self.workflow["jobs"]["diagnose"]["steps"]
            if step.get("id") == "diagnose"
        )
        self.namespace = {"__name__": "diagnostics_test"}
        exec(compile(step["run"], str(WORKFLOW_PATH), "exec"), self.namespace)
        self.environment = {
            "OPENAI_BASE_URL": "https://gateway.example/v1/",
            "OPENAI_API_KEY": "private-test-api-key",
            "LITELLM_MODEL": "openai/provider/model-id",
            "PROBE_MODEL": "true",
        }

    def test_manual_only_read_only_and_generation_opt_in(self):
        self.assertEqual(set(self.workflow["on"]), {"workflow_dispatch"})
        self.assertEqual(
            self.workflow["on"]["workflow_dispatch"]["inputs"]["probe_model"]["default"], "false"
        )
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        self.assertEqual(self.workflow["jobs"]["diagnose"]["timeout-minutes"], "4")

    def test_redirects_are_not_followed(self):
        self.assertIsNone(
            self.namespace["NoRedirect"]().redirect_request(None, None, 302, "", {}, "https://other.example")
        )

    def test_rejects_unsafe_urls_before_network_access(self):
        for base in (
            "http://gateway.example", "https://user:password@gateway.example",
            "https://gateway.example?key=private", "https://gateway.example/#fragment",
            "https://gateway example",
        ):
            with (
                self.subTest(base=base),
                patch.dict(os.environ, {**self.environment, "OPENAI_BASE_URL": base}, clear=True),
            ):
                probe = MagicMock()
                with patch.dict(self.namespace, {"probe": probe}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.namespace["main"](), 1)
                probe.assert_not_called()

    def test_preserves_gateway_alias_and_bounds_generation(self):
        probe = MagicMock(side_effect=[
            (401, None), (200, {"data": [{"id": "provider/model-id"}]}), (200, {"choices": [{}]})
        ])
        with (
            patch.dict(os.environ, self.environment, clear=True),
            patch("socket.getaddrinfo", return_value=[]),
            patch.dict(self.namespace, {"probe": probe}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(self.namespace["main"](), 0)
        self.assertEqual(probe.call_count, 3)
        request = probe.call_args_list[2]
        self.assertEqual(request.args[1], "https://gateway.example/v1/chat/completions")
        self.assertEqual(request.kwargs["payload"]["model"], "provider/model-id")
        self.assertEqual(request.kwargs["payload"]["max_tokens"], 32)
        self.assertFalse(request.kwargs["payload"]["stream"])
        self.assertEqual(request.kwargs["timeout"], 60)

    def test_authentication_failure_skips_generation(self):
        probe = MagicMock(side_effect=[(401, None), (401, None)])
        with (
            patch.dict(os.environ, self.environment, clear=True),
            patch("socket.getaddrinfo", return_value=[]),
            patch.dict(self.namespace, {"probe": probe}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(self.namespace["main"](), 1)
        self.assertEqual(probe.call_count, 2)

    def test_generation_requires_opt_in(self):
        probe = MagicMock(side_effect=[(401, None), (200, {"data": []})])
        with (
            patch.dict(os.environ, {**self.environment, "PROBE_MODEL": "false"}, clear=True),
            patch("socket.getaddrinfo", return_value=[]),
            patch.dict(self.namespace, {"probe": probe}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(self.namespace["main"](), 0)
        self.assertEqual(probe.call_count, 2)

    def test_network_exceptions_do_not_print_sensitive_details(self):
        opener = MagicMock()
        opener.open.side_effect = URLError(TimeoutError("private-test-api-key"))
        output = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(self.namespace, {"build_opener": MagicMock(return_value=opener)}),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                self.namespace["probe"]("Network", "https://gateway.example/models"), (0, None)
            )
        self.assertIn("TimeoutError", output.getvalue())
        self.assertNotIn("private-test-api-key", output.getvalue())

    def test_http_errors_and_success_bodies_are_not_logged(self):
        for response in (
            HTTPError(
                "https://gateway.example", 401, "private-test-api-key", {}, io.BytesIO(b"private-test-api-key")
            ),
            None,
        ):
            opener = MagicMock()
            if response is not None:
                opener.open.side_effect = response
            else:
                body = {"data": [{"id": "private-test-api-key"}]}
                opener.open.return_value.__enter__.return_value.status = 200
                opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()
            output = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(self.namespace, {"build_opener": MagicMock(return_value=opener)}),
                contextlib.redirect_stdout(output),
            ):
                self.namespace["probe"]("Auth", "https://gateway.example/models")
            self.assertNotIn("private-test-api-key", output.getvalue())

    def test_model_failure_is_not_reported_as_success(self):
        probe = MagicMock(side_effect=[(401, None), (200, {"data": []}), (503, None)])
        output = io.StringIO()
        with (
            patch.dict(os.environ, self.environment, clear=True),
            patch("socket.getaddrinfo", return_value=[]),
            patch.dict(self.namespace, {"probe": probe}),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(self.namespace["main"](), 1)
        self.assertNotIn("PASS: minimal model response", output.getvalue())


if __name__ == "__main__":
    unittest.main()
