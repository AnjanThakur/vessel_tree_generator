"""Dependency-free local web application for presenting and verifying Coronary4D."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .live_demo import generate_runtime_case
from .presentation_data import output_contract, phase_geometry, presentation_summary
from .verification import run_fast_verification, run_full_verification


CENTER = Path(__file__).resolve().parent


class PresentationHandler(SimpleHTTPRequestHandler):
    server_version = "Coronary4DPresentation/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, relative: str) -> None:
        candidate = (CENTER / relative).resolve()
        if CENTER.resolve() not in candidate.parents and candidate != CENTER.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/summary":
                self._json(presentation_summary())
            elif parsed.path == "/api/output-contract":
                self._json(output_contract())
            elif parsed.path == "/api/phase":
                query = parse_qs(parsed.query)
                self._json(phase_geometry(query.get("case", ["focal_lad"])[0], int(query.get("index", [0])[0])))
            elif parsed.path == "/api/verify":
                mode = parse_qs(parsed.query).get("mode", ["fast"])[0]
                self._json(run_full_verification() if mode == "full" else run_fast_verification())
            elif parsed.path in {"/", "/index.html"}:
                self._serve_file("index.html")
            else:
                self._serve_file(parsed.path.lstrip("/"))
        except Exception as exc:
            self._json({"status": "FAIL", "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100_000:
                raise ValueError("request too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json(generate_runtime_case(payload))
        except Exception as exc:
            self._json({"status": "FAIL", "error": str(exc)}, HTTPStatus.BAD_REQUEST)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    server = ThreadingHTTPServer((host, port), PresentationHandler)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Coronary4D Presentation & Verification Center: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    return serve(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
