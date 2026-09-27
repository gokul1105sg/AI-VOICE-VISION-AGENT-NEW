"""Shared fixtures for the browser test suite.

Fixtures are served over plain HTTP on an ephemeral 127.0.0.1 port so the
Playwright manager enforces its normal http/https policy while tests stay
hermetic (no public internet).
"""

from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import pytest_asyncio

from browser.manager import PlaywrightManager

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_server() -> str:
    """Serve tests/fixtures over HTTP on an ephemeral loopback port.

    Files under ``/downloads/`` are served with ``Content-Disposition:
    attachment`` so Playwright sees them as downloads.
    """

    class DownloadHandler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            if self.path.startswith("/downloads/"):
                self.send_header("Content-Disposition", "attachment")
                self.send_header("Content-Type", "application/octet-stream")
            super().end_headers()

    handler = functools.partial(DownloadHandler, directory=str(FIXTURES_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest_asyncio.fixture
async def browser(fixture_server, monkeypatch):
    """A started PlaywrightManager, skipped cleanly when Chromium is missing."""
    # Keep the suite headless and quiet regardless of the agent's default
    # (the agent runs BROWSER_HEADLESS=0 so the user can watch it drive pages).
    monkeypatch.setenv("BROWSER_HEADLESS", "1")
    mgr = PlaywrightManager()
    try:
        await mgr.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        await mgr.close()
        pytest.skip(f"Chromium is not available: {exc}")
    yield mgr
    await mgr.close()
