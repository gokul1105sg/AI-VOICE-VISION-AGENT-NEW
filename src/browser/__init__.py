"""Playwright browser control for the Jarvis agent.

This package gives Jarvis a Playwright-managed Chromium browser through two
merged function tools (``browser_action`` and ``browser_query``), powered by
:class:`browser.manager.PlaywrightManager`. The browser starts lazily on the
first tool call and is torn down when the call ends, so every call gets a
fresh ephemeral profile.
"""
