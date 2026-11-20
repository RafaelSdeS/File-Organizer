"""End-to-end clustering quality with the real embedding model.

Slow (loads intfloat/multilingual-e5-small, a ~470 MB download the first
time), so skipped unless RUN_QUALITY=1. Runs the analyzer with its default
settings on examples/demo and compares the folders it proposes with the
known topics in examples/demo_topics.tsv. The unit tests mock the model, so
this is what catches regressions like "pt+en folders split by language".

It also proves the "local" claim: once the model is cached, building the
analyzer and organizing the demo must not touch the network at all.
"""
import csv
import os
import socket
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEMO = os.path.join(ROOT, "examples", "demo")
TOPICS = os.path.join(ROOT, "examples", "demo_topics.tsv")
# Set just below the measured value, so it fails on a real regression but
# not on numeric noise across platforms.
ARI_FLOOR = 0.8
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def offline_guard(attempts):
    """Two patches that refuse any non-loopback DNS lookup or connection and
    record it in `attempts`. Recording matters: huggingface_hub falls back to
    its cache when the network fails, so a refused request alone would pass
    silently."""
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def connect(sock, address):
        if sock.family != getattr(socket, "AF_UNIX", None) and address[0] not in LOOPBACK:
            attempts.append(address)
            raise OSError(f"offline test: blocked connection to {address}")
        return real_connect(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in LOOPBACK:
            attempts.append(host)
            raise socket.gaierror(f"offline test: blocked lookup of {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    return patch.object(socket.socket, "connect", connect), patch.object(socket, "getaddrinfo", getaddrinfo)


@unittest.skipUnless(os.environ.get("RUN_QUALITY") == "1", "set RUN_QUALITY=1 to run (loads the real model)")
class TestDemoQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.document_analyzer.analyzer import DocumentAnalyzer, _load_model

        with open(TOPICS, encoding="utf-8") as f:
            cls.truth = {row["file"]: row for row in csv.DictReader(f, delimiter="\t")}
        _load_model()  # the first run ever downloads the model: the only time the network is allowed
        cls.network_attempts = []
        block_connect, block_dns = offline_guard(cls.network_attempts)
        cls.cache_dir = tempfile.TemporaryDirectory()  # encode for real, not from the user's cache
        no_cache = patch("src.document_analyzer.analyzer.CACHE_PATH", os.path.join(cls.cache_dir.name, "cache.sqlite"))
        with block_connect, block_dns, no_cache:
            cls.analyzer = DocumentAnalyzer()  # defaults: what a user gets with no flags
            cls.folders = cls.analyzer.analyze_directory(DEMO)
        cls.folder_of = {path: folder for folder, paths in cls.folders.items() for path in paths}

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir.cleanup()

    def test_runs_without_network(self):
        self.assertEqual(self.network_attempts, [])

    def test_offline_guard_catches_requests(self):
        """Without this, a guard that silently stopped working would make
        test_runs_without_network pass for the wrong reason."""
        attempts = []
        block_connect, block_dns = offline_guard(attempts)
        with block_connect, block_dns, self.assertRaises(OSError):
            urllib.request.urlopen("https://huggingface.co", timeout=5)
        self.assertTrue(attempts)

    def test_folders_match_topics(self):
        from sklearn.metrics import adjusted_rand_score

        files = sorted(self.truth)
        # Each one-off is its own topic: grouping two of them is a mistake.
        topics = [self.truth[f]["topic"] if self.truth[f]["topic"] != "one-off" else f for f in files]
        ari = adjusted_rand_score(topics, [self.folder_of[f] for f in files])
        self.assertGreaterEqual(ari, ARI_FLOOR, f"folders: {self.folders}")

    def test_folders_are_not_split_by_language(self):
        """Every topic mixes pt and en, so any sizeable folder should too."""
        for folder, paths in self.folders.items():
            if len(paths) >= 4:
                languages = {self.truth[p]["language"] for p in paths}
                self.assertEqual(languages, {"pt", "en"}, f"{folder} is monolingual: {paths}")

    def test_report_renders(self):
        from src.document_analyzer.report import save_report

        with tempfile.TemporaryDirectory() as out_dir:
            saved = save_report(self.analyzer.last_run, self.folders, out_dir)
            self.assertEqual(sorted(os.path.basename(p) for p in saved),
                             ["k_selection.png", "map.png", "similarity.png"])


if __name__ == "__main__":
    unittest.main()
