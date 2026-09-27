"""Environment-driven configuration for the browser tools.

All settings are read from the environment on each access so tests can
monkeypatch them without restarting anything. See the README's browser section
for a table of these variables.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

_DEFAULT_DOMAINS = ""
_DEFAULT_HEADLESS = "0"  # visible browser by default
_DEFAULT_TIMEOUT_MS = "15000"

_FALSY = {"", "0", "false", "no", "off"}


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSY


def headless() -> bool:
    """Whether Chromium runs headless (``BROWSER_HEADLESS``, default ``0``)."""
    return _flag("BROWSER_HEADLESS", _DEFAULT_HEADLESS)


def no_sandbox() -> bool:
    """Launch Chromium with ``--no-sandbox`` (``BROWSER_NO_SANDBOX``)."""
    return _flag("BROWSER_NO_SANDBOX", "0")


def allow_js() -> bool:
    """Enable the ``run_javascript`` action (``BROWSER_ALLOW_JS``).

    Even when enabled, the ``run_javascript`` browser action still requires
    ``confirmed=True`` from the calling agent.
    """
    return _flag("BROWSER_ALLOW_JS", "0")


def timeout_ms() -> int:
    """Default per-action/navigation timeout in milliseconds."""
    raw = os.environ.get("BROWSER_TIMEOUT_MS", _DEFAULT_TIMEOUT_MS).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return int(_DEFAULT_TIMEOUT_MS)


def allowed_domains() -> list[str]:
    """Comma-separated allowlist (``BROWSER_ALLOWED_DOMAINS``).

    Empty means any ``http(s)`` host is allowed.
    """
    raw = os.environ.get("BROWSER_ALLOWED_DOMAINS", _DEFAULT_DOMAINS)
    return [d.strip().lower().lstrip(".") for d in raw.split(",") if d.strip()]


def is_allowed_url(url: str) -> bool:
    """True for ``http``/``https`` URLs on an allowed host.

    When ``BROWSER_ALLOWED_DOMAINS`` is set, only exact hosts or subdomains of
    the listed domains pass.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    domains = allowed_domains()
    if not domains:
        return True
    host = parsed.hostname.lower()
    return any(host == d or host.endswith("." + d) for d in domains)
