"""Hermetic tests for the Playwright browser tools.

The ``browser`` fixture (tests/conftest.py) starts a real
``PlaywrightManager`` against local fixture pages served over 127.0.0.1, so no
external internet is needed. Tests that need Chromium are tagged ``browser``
and are skipped automatically when the runtime is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from livekit.agents.llm import ToolContext, ToolError

from browser.manager import PlaywrightManager
from browser.tools import browser_action, browser_query, require_confirmation

# -- pure checks (no Chromium needed) ----------------------------------------


def test_browser_tools_are_function_tools() -> None:
    assert browser_action.id == "browser_action"
    assert browser_query.id == "browser_query"


def test_browser_tool_schemas_expose_actions() -> None:
    schemas = ToolContext(tools=[browser_action, browser_query]).parse_function_tools(
        format="openai"
    )
    by_name = {s["function"]["name"]: s["function"] for s in schemas}

    action_enum = by_name["browser_action"]["parameters"]["properties"]["action"][
        "enum"
    ]
    for action in (
        "open_url",
        "click",
        "type_text",
        "open_new_tab",
        "switch_tab",
        "screenshot",
        "get_cookies",
        "clear_cookies",
        "upload_file",
        "download_file",
        "run_javascript",
    ):
        assert action in action_enum

    query_enum = by_name["browser_query"]["parameters"]["properties"]["action"]["enum"]
    for action in (
        "inspect_page",
        "read_page",
        "get_url",
        "get_title",
        "list_tabs",
        "extract_links",
    ):
        assert action in query_enum


def test_require_confirmation_raises_without_confirm() -> None:
    require_confirmation(True, "run_javascript")  # no raise
    with pytest.raises(ToolError):
        require_confirmation(False, "run_javascript")


# -- navigation & reading ----------------------------------------------------


@pytest.mark.browser
async def test_navigate_and_read(
    browser: PlaywrightManager, fixture_server: str
) -> None:
    opened = await browser.navigate(fixture_server + "/index.html")
    assert opened["url"].endswith("index.html")
    assert opened["title"] == "Jarvis Test Home"

    assert "Welcome to the test page" in await browser.read_page()
    assert (await browser.get_url()).endswith("index.html")
    assert await browser.get_title() == "Jarvis Test Home"


@pytest.mark.browser
async def test_read_page_with_selector(
    browser: PlaywrightManager, fixture_server: str
) -> None:
    await browser.navigate(fixture_server + "/index.html")
    assert (await browser.read_page(selector="#result")) == "original"
    with pytest.raises(ToolError):
        await browser.read_page(selector="#does-not-exist")


@pytest.mark.browser
async def test_extract_links(browser: PlaywrightManager, fixture_server: str) -> None:
    await browser.navigate(fixture_server + "/links.html")
    links = await browser.extract_links()
    urls = {link["url"] for link in links}
    assert "https://example.com/one" in urls
    assert "https://example.com/two" in urls


# -- interaction -------------------------------------------------------------


@pytest.mark.browser
async def test_inspect_type_click(
    browser: PlaywrightManager, fixture_server: str
) -> None:
    await browser.navigate(fixture_server + "/index.html")

    inventory = await browser.inspect_page()
    assert "Click me" in inventory
    assert "Your name" in inventory

    # role:name targeting
    await browser.type_text("textbox:Your name", "sir")
    await browser.click("button:Click me")
    assert (await browser.read_page(selector="#result")) == "Hello sir"

    # numbered-index targeting: the button is the first interactive element
    await browser.click("1")
    assert (await browser.read_page(selector="#result")) == "Hello sir"

    # reload clears the form state
    await browser.reload()
    await browser.click("1")
    assert (await browser.read_page(selector="#result")) == "Hello world"


@pytest.mark.browser
async def test_select_option(browser: PlaywrightManager, fixture_server: str) -> None:
    await browser.navigate(fixture_server + "/select.html")
    await browser.select_option("combobox:Color", "green")
    assert (await browser.read_page(selector="#out")) == "green"
    await browser.select_option("#c", "blue")  # CSS selector targeting
    assert (await browser.read_page(selector="#out")) == "blue"


@pytest.mark.browser
async def test_form_type_and_submit(
    browser: PlaywrightManager, fixture_server: str
) -> None:
    await browser.navigate(fixture_server + "/form.html")
    await browser.type_text("textbox:Email", "sir@example.com")
    await browser.click("button:Subscribe")
    assert (await browser.read_page(selector="#out")) == "submitted sir@example.com"


@pytest.mark.browser
async def test_clear_field_and_press_key(
    browser: PlaywrightManager, fixture_server: str
) -> None:
    await browser.navigate(fixture_server + "/index.html")
    await browser.type_text("textbox:Your name", "sir")
    await browser.clear_field("textbox:Your name")
    await browser.press_key("Escape")  # still interactive afterwards
    await browser.click("button:Click me")
    assert (await browser.read_page(selector="#result")) == "Hello world"


@pytest.mark.browser
async def test_scroll(browser: PlaywrightManager, fixture_server: str) -> None:
    await browser.navigate(fixture_server + "/tall.html")
    page = browser.active_page
    bottom = page.locator("p:has-text('bottom')")

    async def in_viewport() -> bool:
        return bool(
            await bottom.evaluate(
                "(el) => el.getBoundingClientRect().top < window.innerHeight"
            )
        )

    assert await in_viewport() is False
    await browser.scroll(direction="down", amount=3000)
    assert await in_viewport() is True
    await browser.scroll(direction="up", amount=3000)
    assert await in_viewport() is False
    with pytest.raises(ToolError):
        await browser.scroll(direction="sideways")


# -- tabs --------------------------------------------------------------------


@pytest.mark.browser
async def test_tabs(browser: PlaywrightManager, fixture_server: str) -> None:
    await browser.navigate(fixture_server + "/index.html")
    opened = await browser.open_new_tab(fixture_server + "/links.html")
    assert opened["tab"] == 1
    assert (await browser.get_url()).endswith("links.html")

    tabs = await browser.list_tabs()
    assert len(tabs) == 2
    assert tabs[1]["active"] is True

    switched = await browser.switch_tab(0)
    assert switched["tab"] == 0
    assert (await browser.get_url()).endswith("index.html")

    closed = await browser.close_tab(0)
    assert closed["tabs_open"] == 1
    assert closed["active_tab"] == 0
    assert (await browser.get_url()).endswith("links.html")


# -- screenshots / cookies ---------------------------------------------------


@pytest.mark.browser
async def test_screenshot(browser: PlaywrightManager, fixture_server: str) -> None:
    await browser.navigate(fixture_server + "/index.html")
    assert Path(await browser.screenshot()).is_file()
    assert Path(await browser.screenshot(full_page=True)).is_file()


@pytest.mark.browser
async def test_cookies_get_and_clear(
    browser: PlaywrightManager, fixture_server: str
) -> None:
    await browser.navigate(fixture_server + "/index.html")
    await browser.active_page.evaluate(
        "() => { document.cookie = 'hello=world; path=/'; }"
    )
    cookies = await browser.get_cookies()
    assert any(c["name"] == "hello" and c["domain"] == "127.0.0.1" for c in cookies)

    cleared = await browser.clear_cookies()
    assert "Cleared" in cleared
    assert not any(c["name"] == "hello" for c in await browser.get_cookies())


# -- uploads / downloads -----------------------------------------------------


@pytest.mark.browser
async def test_upload_file(
    browser: PlaywrightManager, fixture_server: str, tmp_path: Path
) -> None:
    payload = tmp_path / "cv.txt"
    payload.write_text("hello", encoding="utf-8")
    await browser.navigate(fixture_server + "/upload.html")
    await browser.upload_file("#file", str(payload))
    assert (await browser.read_page(selector="#out")) == "selected cv.txt"
    with pytest.raises(ToolError):
        await browser.upload_file("#file", str(tmp_path / "missing.txt"))


@pytest.mark.browser
async def test_download_file(browser: PlaywrightManager, fixture_server: str) -> None:
    await browser.navigate(fixture_server + "/download.html")
    dest = await browser.download(target="#dl")
    saved = Path(dest)
    assert saved.name == "hello.txt"
    assert saved.read_text(encoding="utf-8").strip() == "hello from jarvis fixtures"


# -- guardrails --------------------------------------------------------------


@pytest.mark.browser
async def test_navigation_restrictions(
    browser: PlaywrightManager, fixture_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await browser.navigate(fixture_server + "/index.html")  # works by default

    with pytest.raises(ToolError):
        await browser.navigate("ftp://example.com/file")  # non-http(s)
    with pytest.raises(ToolError):
        await browser.navigate("file:///etc/passwd")

    monkeypatch.setenv("BROWSER_ALLOWED_DOMAINS", "127.0.0.1")
    assert (await browser.navigate(fixture_server + "/links.html"))["url"].endswith(
        "links.html"
    )
    with pytest.raises(ToolError):
        await browser.navigate("https://example.com/")


@pytest.mark.browser
async def test_run_javascript_requires_env(
    browser: PlaywrightManager, fixture_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await browser.navigate(fixture_server + "/index.html")
    monkeypatch.delenv("BROWSER_ALLOW_JS", raising=False)
    with pytest.raises(ToolError):
        await browser.run_javascript("1 + 1")

    monkeypatch.setenv("BROWSER_ALLOW_JS", "1")
    assert await browser.run_javascript("1 + 1") == "2"
    assert (await browser.run_javascript("document.title")) == "Jarvis Test Home"
