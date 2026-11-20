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
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files

from .analyzer import organize_files, read_history, undo_organize
from .plan import validate_folders

logger = logging.getLogger(__name__)

CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
MAX_BODY = 10 * 1024 * 1024


def make_server(analyzer, path=None, into=None, target_dir=None, copy=False, port=0):
    """(server, url). The analyzer (and its model) is loaded once and shared
    by every request; the server is single-threaded, so requests never overlap."""
    token = secrets.token_urlsafe(24)
    page = (files(__package__) / "ui.html").read_bytes()
    # The last analysis: /api/apply only ever moves files out of its source.
    state = {"defaults": {"path": path or "", "into": into or "", "copy": copy}, "source": None, "target": None}

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
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                handler = {"/api/analyze": self._analyze, "/api/apply": self._apply, "/api/undo": self._undo}.get(self.path)
                if handler is None:
                    return self._send(404, {"error": "not found"})
                return self._send(200, handler(body))
            except Exception as e:  # shown on the page; the server keeps running
                logger.error(f"ui: {self.path} failed: {e}")
                return self._send(400, {"error": str(e)})

        def _analyze(self, body):
            source = os.path.expanduser(body.get("path", "").strip())
            into = os.path.expanduser(body.get("into", "").strip()) or None
            folders = analyzer.analyze_directory(source, into=into)
            state["source"], state["target"] = source, into or target_dir or source
            last_run = analyzer.last_run
            return {
                "source": os.path.abspath(source),
                "target": os.path.abspath(state["target"]),
                "folders": folders,
                "scores": last_run["scores"],
                "sensitive": last_run["sensitive"],
                "map": _map_points(last_run["paths"], last_run["embeddings"]),
            }

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

    server = HTTPServer(("127.0.0.1", port), Handler)
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


def serve(analyzer, path=None, into=None, target_dir=None, copy=False):
    server, url = make_server(analyzer, path=path, into=into, target_dir=target_dir, copy=copy)
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
