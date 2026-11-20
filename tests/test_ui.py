import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_analyzer.ui import make_server

UI_HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "document_analyzer", "ui.html")


class TestUi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = self.tmp.name
        open(os.path.join(self.source, "a.txt"), "w").close()

        self.analyzer = MagicMock()
        self.analyzer.analyze_directory.return_value = {"Topic": ["a.txt"]}
        self.analyzer.last_run = {"paths": ["a.txt"], "embeddings": [[1.0]], "scores": {"a.txt": 1.0}, "sensitive": {}}

        self.server, url = make_server(self.analyzer, path=self.source)
        self.port = self.server.server_address[1]
        self.token = url.split("#", 1)[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, token=True, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Token"] = self.token
        if host:
            headers["Host"] = host
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data) if response.getheader("Content-Type") == "application/json" else data

    def test_page_is_served_with_a_strict_content_security_policy(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/")
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertIn("default-src 'none'", response.getheader("Content-Security-Policy"))

    def test_page_loads_nothing_from_outside(self):
        with open(UI_HTML, encoding="utf-8") as f:
            html = f.read()
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_api_needs_the_token(self):
        self.assertEqual(self.request("GET", "/api/defaults", token=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/analyze", {"path": self.source}, token=False)[0], 403)
        self.analyzer.analyze_directory.assert_not_called()

    def test_api_refuses_a_foreign_host_header(self):
        """DNS rebinding: a page on evil.example resolving to 127.0.0.1."""
        self.assertEqual(self.request("GET", "/api/defaults", host=f"evil.example:{self.port}")[0], 403)
        self.assertEqual(self.request("GET", "/", host="evil.example")[0], 403)

    def test_analyze_then_apply(self):
        status, result = self.request("POST", "/api/analyze", {"path": self.source, "into": ""})
        self.assertEqual(status, 200)
        self.assertEqual(result["folders"], {"Topic": ["a.txt"]})
        self.analyzer.analyze_directory.assert_called_once_with(self.source, into=None)

        with patch("src.document_analyzer.ui.organize_files", return_value="m.json") as organize:
            status, result = self.request("POST", "/api/apply", {"folders": {"Renamed": ["a.txt"], "Empty": []}, "copy": True})
        self.assertEqual((status, result), (200, {"manifest": "m.json"}))
        organize.assert_called_once_with({"Renamed": ["a.txt"]}, self.source, self.source, copy=True)

    def test_apply_validates_user_edits(self):
        self.request("POST", "/api/analyze", {"path": self.source})
        with patch("src.document_analyzer.ui.organize_files") as organize:
            status, result = self.request("POST", "/api/apply", {"folders": {"../outside": ["a.txt"]}})
        self.assertEqual(status, 400)
        self.assertIn("Invalid folder name", result["error"])
        organize.assert_not_called()

    def test_apply_needs_an_analysis_first(self):
        status, result = self.request("POST", "/api/apply", {"folders": {"X": ["a.txt"]}})
        self.assertEqual(status, 400)

    def test_undo_only_accepts_recorded_manifests(self):
        with patch("src.document_analyzer.ui.read_history", return_value=[]), \
             patch("src.document_analyzer.ui.undo_organize") as undo:
            status, _ = self.request("POST", "/api/undo", {"manifest": "/etc/anything.json"})
        self.assertEqual(status, 400)
        undo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
