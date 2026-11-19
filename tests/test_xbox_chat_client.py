# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT

"""Exercise the Mac proxy without downloading models or contacting an Xbox."""

import importlib.util
import io
import json
import secrets
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("xbox_chat", ROOT / "scripts/xbox-chat.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


class StubXbox:
    def __init__(self):
        self.calls = []
        self.chat = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"completion_tokens":1}}\n\n'
            'data: [DONE]\n\n'
        )
        self.pull = '{"status":"pulling model","completed":1,"total":2}\n{"status":"success"}\n'

    def open(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        payload = None if isinstance(request, str) else json.loads(request.data)
        self.calls.append((url, payload))
        if url.endswith("/v1/models"):
            return io.BytesIO(json.dumps({"data": [{"id": "gemma3-270m"}, {"id": "unapproved"}]}).encode())
        if url.endswith("/api/pull"):
            return io.BytesIO(self.pull.encode())
        return io.BytesIO(self.chat.encode())


class XboxChatClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = CLIENT.ThreadingHTTPServer(("127.0.0.1", 0), CLIENT.Handler)
        self.server.token = secrets.token_urlsafe(16)
        self.server.gate = threading.Lock()
        self.server.base = "http://stub-xbox:11434"
        self.server.opener = StubXbox()
        self.server.log_path = Path(self.temp.name) / "runs.jsonl"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def post(self, path, payload, token=True):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Local-Token"] = self.server.token
        request = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                         headers=headers)
        with self.opener.open(request, timeout=3) as response:
            return [json.loads(line) for line in response if line.strip()]

    def chat(self, **changes):
        value = {"model": "ministral3-3b-instruct", "messages": [{"role": "user", "content": "Hi"}],
                 "max_tokens": 128}
        value.update(changes)
        return value

    def test_selected_model_reaches_xbox_and_stream_is_logged(self):
        events = self.post("/chat", self.chat())
        self.assertEqual(events[0]["delta"], "Hello")
        self.assertTrue(events[-1]["done"])
        self.assertEqual(self.server.opener.calls[0][1]["model"], "ministral3-3b-instruct")
        self.assertTrue(self.server.opener.calls[0][1]["stream"])
        record = json.loads(self.server.log_path.read_text())
        self.assertIsNone(record["error"])
        self.assertEqual(record["model"], "ministral3-3b-instruct")

    def test_long_and_uncapped_replies_reach_xbox(self):
        long_request = self.chat(max_tokens=4096, xbox_experiment="baseline")
        self.post("/chat", long_request)
        forwarded = self.server.opener.calls[-1][1]
        self.assertEqual(forwarded["max_tokens"], 4096)
        self.assertEqual(forwarded["xbox_experiment"], "baseline")
        auto_request = self.chat()
        del auto_request["max_tokens"]
        self.post("/chat", auto_request)
        self.assertNotIn("max_tokens", self.server.opener.calls[-1][1])
        for invalid in (self.chat(max_tokens=-1), self.chat(xbox_experiment="arbitrary")):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post("/chat", invalid)
            self.assertEqual(error.exception.code, 400)

    def test_unknown_model_and_image_content_are_rejected_before_xbox_call(self):
        for payload in [self.chat(model="unapproved"),
                        self.chat(messages=[{"role": "user", "content": [{"type": "image"}]}])]:
            with self.subTest(payload=payload):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.post("/chat", payload)
                self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.server.opener.calls, [])

    def test_missing_local_token_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post("/pull", {"model": "ministral3-3b-instruct"}, token=False)
        self.assertEqual(error.exception.code, 403)
        self.assertEqual(self.server.opener.calls, [])

    def test_pull_streams_progress_and_requires_success(self):
        events = self.post("/pull", {"model": "ministral3-3b-instruct"})
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(self.server.opener.calls[0][0], self.server.base + "/api/pull")
        self.server.opener.pull = '{"status":"pulling model"}\n'
        events = self.post("/pull", {"model": "ministral3-3b-instruct"})
        self.assertIn("without success", events[-1]["error"])
        self.assertFalse(self.server.gate.locked())

    def test_incomplete_generation_is_an_error_not_a_success(self):
        self.server.opener.chat = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        events = self.post("/chat", self.chat())
        self.assertIn("without its completion marker", events[-1]["error"])
        self.assertFalse(any(event.get("done") for event in events))
        self.assertIsNotNone(json.loads(self.server.log_path.read_text())["error"])

    def test_discovery_and_client_model_policy_match_catalogue(self):
        with self.opener.open(self.base + "/models", timeout=3) as response:
            models = json.load(response)["models"]
        self.assertEqual({model["id"] for model in models}, set(CLIENT.MODELS))
        self.assertEqual([model["id"] for model in models if model["installed"]], ["gemma3-270m"])
        catalogue = json.loads((ROOT / "uwp/models/manifest.xbox-one.json").read_text())
        self.assertEqual(set(CLIENT.MODELS), {model["name"] for model in catalogue["models"]})


if __name__ == "__main__":
    unittest.main()
