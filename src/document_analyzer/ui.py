"""--ui: the same pipeline behind a page in your browser.

Served on 127.0.0.1 only, by the stdlib. The page (ui.html) is one file with
no external URL, and the Content-Security-Policy below makes the browser
refuse to load anything from anywhere else. Every /api/ call needs the
per-run token from the URL's #fragment (fragments are never sent in
requests, so it can't leak into logs) and a Host header naming this server,
so neither another website nor a DNS-rebinding trick can drive it.
"""
import json
import logging
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from .analyzer import _read_content, organize_files, read_history, undo_organize
from .plan import parse_folders, validate_folders

logger = logging.getLogger(__name__)

CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
MAX_BODY = 10 * 1024 * 1024


class _Progress(logging.Handler):
    """Keeps the package's latest INFO line ("Read 300 entries...") for
    /api/progress. ponytail: stage-level only; the embedding stage just says
    "generating embeddings..." - log per batch if that stage feels stuck."""

    def __init__(self, state):
        super().__init__(logging.INFO)
        self.state = state

    def emit(self, record):
        self.state["progress"] = record.getMessage()


def make_server(analyzer, path=None, into=None, target_dir=None, copy=False, folders=None, port=0):
    """(server, url). The analyzer (and its model) is loaded once and shared
    by every request. Requests get a thread each, so /api/progress answers
    while an analysis runs, but the POSTs hold a lock: one at a time, never
    overlapping on the analyzer or the state below."""
    token = secrets.token_urlsafe(24)
    page = (files(__package__) / "ui.html").read_bytes()
    folders_text = ", ".join(folders) if isinstance(folders, list) else str(folders or "")
    # The last analysis: /api/apply only ever moves files out of its source,
    # and /api/preview and /api/split only take files it listed.
    state = {"defaults": {"path": path or "", "into": into or "", "copy": copy, "folders": folders_text},
             "source": None, "target": None, "paths": set(), "progress": ""}
    lock = threading.Lock()
    logging.getLogger(__package__).addHandler(_Progress(state))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug("ui: " + format % args)

        def _host_ok(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            if self._host_ok() and secrets.compare_digest(self.headers.get("X-Token", ""), token):
                return True
            self._send(403, {"error": "forbidden"})
            return False

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "forbidden"})
            if self.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if not self.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return None
            if self.path == "/api/defaults":
                return self._send(200, state["defaults"])
            if self.path == "/api/history":
                return self._send(200, [dict(e, undone=not os.path.exists(e.get("manifest", ""))) for e in read_history()])
            if self.path == "/api/progress":
                return self._send(200, {"message": state["progress"]})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                handler = {"/api/analyze": self._analyze, "/api/apply": self._apply, "/api/undo": self._undo,
                           "/api/preview": self._preview, "/api/split": self._split}.get(self.path)
                if handler is None:
                    return self._send(404, {"error": "not found"})
                with lock:
                    result = handler(body)
                return self._send(200, result)
            except Exception as e:  # shown on the page; the server keeps running
                logger.error(f"ui: {self.path} failed: {e}")
                return self._send(400, {"error": str(e)})

        def _analyze(self, body):
            source = os.path.expanduser(body.get("path", "").strip())
            into = os.path.expanduser(body.get("into", "").strip()) or None
            state["progress"] = ""
            folders = analyzer.analyze_directory(source, into=into, folders=parse_folders(body.get("folders")))
            state["source"], state["target"] = source, into or target_dir or source
            last_run = analyzer.last_run
            state["paths"] = set(last_run["paths"])
            k_scores = last_run["k_scores"]  # None when the user fixed the folders
            return {
                "source": os.path.abspath(source),
                "target": os.path.abspath(state["target"]),
                "folders": folders,
                "scores": last_run["scores"],
                "sensitive": last_run["sensitive"],
                "map": _map_points(last_run["paths"], last_run["embeddings"]),
                # The chosen k and each metric's own pick (the chosen one is
                # their median, so not always among them), as buttons.
                "suggest": k_scores and {"chosen": k_scores["chosen"],
                                         "options": sorted({k_scores["chosen"], *(k for k in k_scores["votes"].values() if k)})},
            }

        def _known(self, files):
            if not isinstance(files, list) or not files or not set(files) <= state["paths"]:
                raise ValueError("Unknown file: analyze the folder again")

        def _preview(self, body):
            """The start of a file's text - what the analysis read of it."""
            self._known([body.get("file")])
            return {"text": (_read_content(os.path.join(state["source"], body["file"])) or "")[:600]}

        def _split(self, body):
            name = str(body.get("name") or "").strip()
            if not name:
                raise ValueError("Which folder?")
            self._known(body.get("files"))
            folders, scores = analyzer.split_folder(name, body["files"])
            return {"folders": folders, "scores": scores}

        def _apply(self, body):
            if not state["source"]:
                raise ValueError("Analyze a folder first")
            folders = body.get("folders") or {}
            if not isinstance(folders, dict) or not all(isinstance(v, list) for v in folders.values()):
                raise ValueError("folders must map folder names to lists of files")
            folders = {name: files for name, files in folders.items() if files}
            validate_folders(folders, state["source"])
            manifest = organize_files(folders, state["source"], state["target"], copy=bool(body.get("copy")))
            return {"manifest": manifest}

        def _undo(self, body):
            # Only manifests this tool recorded - not any path a request names.
            manifest = body.get("manifest")
            if manifest not in {e.get("manifest") for e in read_history()}:
                raise ValueError("Unknown manifest")
            undo_organize(manifest)
            return {"complete": not os.path.exists(manifest)}

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, f"http://127.0.0.1:{server.server_address[1]}/#{token}"


def _map_points(paths, embeddings):
    """[{file, x, y}] for the page's map, or [] when there's no map to draw."""
    if len(paths) < 3:
        return []
    from .report import tsne_xy
    try:
        xy = tsne_xy(embeddings)
    except Exception as e:  # the map is a nicety; never fail the analysis for it
        logger.warning(f"ui: no map: {e}")
        return []
    return [{"file": p, "x": float(x), "y": float(y)} for p, (x, y) in zip(paths, xy)]


def serve(analyzer, path=None, into=None, target_dir=None, copy=False, folders=None):
    server, url = make_server(analyzer, path=path, into=into, target_dir=target_dir, copy=copy, folders=folders)
    # flush: piped or logged, stdout is block-buffered and the link (the only
    # way in, it carries the token) would sit unseen while the server runs.
    print(f"File Organizer is open at {url}\n(only on this computer; Ctrl+C to stop)", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
