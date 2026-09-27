"""PlaywrightManager: a Playwright-controlled Chromium browser for the agent.

One :class:`PlaywrightManager` owns one browser context (tabs, cookies,
downloads) and is meant to live for a single agent call/session. Managers are
held in a per-session (``id(session)``) registry so concurrent jobs in one
worker don't share a browser, and are closed when their session ends.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from livekit.agents.llm import ToolError
from playwright.async_api import (
    Browser,
    BrowserContext,
    Download,
    Locator,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from . import config

logger = logging.getLogger(__name__)

MAX_INSPECT_ITEMS = 30
MAX_READ_CHARS = 4000
MAX_LINKS = 25

_BASE_DATA_DIR: Path = Path(tempfile.gettempdir()) / "jarvis-browser"

_DOWNLOAD_DIR = _BASE_DATA_DIR / "downloads"
_SCREENSHOT_DIR = _BASE_DATA_DIR / "screenshots"

# Interactive element groups probed by inspect_page, in priority order.
_GROUPS: list[tuple[str, str]] = [
    ("button", "button, input[type='button'], input[type='submit'], [role='button']"),
    ("link", "a[href]"),
    (
        "textbox",
        "input:not([type='hidden']), textarea, [contenteditable='true'], [role='textbox']",
    ),
    ("combobox", "select, [role='combobox']"),
    (
        "checkable",
        "input[type='checkbox'], input[type='radio'], [role='checkbox'], [role='radio']",
    ),
    ("other", "[tabindex]:not([tabindex='-1']), summary, [onclick], [role='menuitem']"),
]

_ROLES = {
    "button",
    "link",
    "textbox",
    "combobox",
    "checkbox",
    "radio",
    "option",
    "menuitem",
    "tab",
    "heading",
    "img",
    "listbox",
    "switch",
}


def _session_key(session: object) -> int:
    """Stable registry key for a session's browser manager."""
    return id(session)


_managers: dict[int, PlaywrightManager] = {}


async def close_for(session: object) -> None:
    """Close and forget the browser for ``session`` (called on session end)."""
    mgr = _managers.pop(_session_key(session), None)
    if mgr is not None:
        await mgr.close()


def get_manager(session: object) -> PlaywrightManager:
    """Return (creating on first use) the browser manager for ``session``."""
    mgr = _managers.get(_session_key(session))
    if mgr is None:
        mgr = PlaywrightManager()
        _managers[_session_key(session)] = mgr
    return mgr


class PlaywrightManager:
    """Owns a Chromium browser plus its context, tabs, and page state."""

    def __init__(self) -> None:
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: list[Page] = []
        self._active_index = 0
        self._inspect_items: dict[str, Locator] = {}
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """Launch Chromium (lazily, on first use) with a fresh context."""
        if self._started:
            return
        self._pw = await async_playwright().start()
        launch_opts: dict[str, Any] = {"headless": config.headless()}
        if config.no_sandbox():
            launch_opts["args"] = ["--no-sandbox"]
        self._browser = await self._pw.chromium.launch(**launch_opts)
        self._context = await self._browser.new_context(accept_downloads=True)
        page = await self._context.new_page()
        page.set_default_timeout(config.timeout_ms())
        self._pages = [page]
        self._active_index = 0
        self._inspect_items = {}
        self._started = True
        logger.info("Playwright Chromium started (headless=%s)", config.headless())

    async def close(self) -> None:
        """Close the browser and all its state. Safe to call repeatedly."""
        self._started = False
        self._pages.clear()
        self._inspect_items.clear()
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None
        logger.info("Playwright Chromium closed")

    # -- helpers -------------------------------------------------------------

    @property
    def active_page(self) -> Page:
        if not self._started or not self._pages:
            raise ToolError("The browser is not open. Open a URL first.")
        if self._active_index >= len(self._pages):
            self._active_index = len(self._pages) - 1
        return self._pages[self._active_index]

    def _check_url(self, url: str) -> None:
        if not config.is_allowed_url(url):
            domains = config.allowed_domains()
            hint = (
                f" Only these domains are allowed: {', '.join(domains)}."
                if domains
                else ""
            )
            raise ToolError(
                f"Refusing to open {url!r}: only http(s) URLs are supported.{hint}"
            )

    async def _goto(self, page: Page, url: str) -> None:
        self._check_url(url)
        try:
            await page.goto(url, wait_until="load", timeout=config.timeout_ms())
        except PlaywrightTimeoutError as exc:
            raise ToolError(
                f"Timed out after {config.timeout_ms()} ms loading {url}."
            ) from exc

    async def _resolve_target(self, target: str) -> Locator:
        """Resolve a target spec to a Playwright locator.

        Supported forms, tried in order: the number from the last
        ``inspect_page``; ``role:name``; a CSS selector; visible text.
        """
        target = (target or "").strip()
        if not target:
            raise ToolError(
                "A target element is required. Run inspect_page to see available targets."
            )
        if target.isdigit():
            loc = self._inspect_items.get(str(int(target)))
            if loc is None:
                raise ToolError(
                    f"No numbered element {int(target)} from the last inspection. "
                    "Run inspect_page again for fresh targets."
                )
            return loc
        if ":" in target:
            role, _, name = target.partition(":")
            role, name = role.strip().lower(), name.strip()
            if role in _ROLES and name:
                loc = self.active_page.get_by_role(role, name=name, exact=False).first
                if await loc.count():
                    return loc
                raise ToolError(
                    f"No element with role {role!r} and name {name!r} on the page."
                )
        for probe in (
            lambda: self.active_page.locator(target).first,
            lambda: self.active_page.get_by_text(target, exact=False).first,
        ):
            loc = probe()
            if await loc.count():
                return loc
        raise ToolError(
            f"Could not find an element matching {target!r} on the page. "
            "Run inspect_page to see available targets."
        )

    async def _describe(self, el: Locator, role: str) -> str:
        page = self.active_page
        tag = (await el.evaluate("(e) => e.tagName.toLowerCase()")) or "element"
        aria_label = await el.get_attribute("aria-label")
        el_id = await el.get_attribute("id")
        name = await el.get_attribute("name")
        placeholder = await el.get_attribute("placeholder")
        el_type = await el.get_attribute("type")
        try:
            text = re.sub(r"\s+", " ", (await el.inner_text()).strip())[:60]
        except PlaywrightError:
            text = ""
        label = (
            aria_label
            or (text[:40] if text else "")
            or placeholder
            or el_id
            or f"<{tag}>"
        )
        bits = [f'{tag} "{label}"']
        href = await el.get_attribute("href")
        if role == "link" and href:
            bits.append("-> " + urljoin(page.url, href))
        details = []
        if el_id and el_id != label:
            details.append(f"#{el_id}")
        if name and name != label:
            details.append(f"name={name}")
        if el_type and el_type not in ("text",):
            details.append(f"type={el_type}")
        if details:
            bits.append("(" + ", ".join(details) + ")")
        return " ".join(bits)

    # -- navigation ----------------------------------------------------------

    async def navigate(self, url: str) -> dict[str, str]:
        page = self.active_page
        await self._goto(page, url)
        return {"url": page.url, "title": await page.title()}

    async def reload(self) -> str:
        page = self.active_page
        try:
            await page.reload(wait_until="load", timeout=config.timeout_ms())
        except PlaywrightTimeoutError as exc:
            raise ToolError("Timed out reloading the page.") from exc
        return f"Reloaded {page.url}."

    async def go_back(self) -> bool:
        page = self.active_page
        if not await page.can_go_back():
            return False
        await page.go_back(wait_until="domcontentloaded", timeout=config.timeout_ms())
        return True

    async def go_forward(self) -> bool:
        page = self.active_page
        if not await page.can_go_forward():
            return False
        await page.go_forward(
            wait_until="domcontentloaded", timeout=config.timeout_ms()
        )
        return True

    # -- tabs ----------------------------------------------------------------

    async def open_new_tab(self, url: str | None = None) -> dict[str, Any]:
        if self._context is None:
            raise ToolError("The browser is not open. Open a URL first.")
        page = await self._context.new_page()
        page.set_default_timeout(config.timeout_ms())
        self._pages.append(page)
        self._active_index = len(self._pages) - 1
        if url:
            await self._goto(page, url)
        return {"tab": self._active_index, "url": page.url, "title": await page.title()}

    async def switch_tab(self, index: int) -> dict[str, str]:
        if not self._pages:
            raise ToolError("The browser is not open. Open a URL first.")
        if not (0 <= index < len(self._pages)):
            raise ToolError(
                f"Tab {index} does not exist. Open tabs: 0..{len(self._pages) - 1}."
            )
        self._active_index = index
        page = self._pages[index]
        return {"tab": index, "url": page.url, "title": await page.title()}

    async def close_tab(self, index: int | None = None) -> dict[str, Any]:
        if not self._pages:
            raise ToolError("The browser is not open. Open a URL first.")
        idx = self._active_index if index is None else index
        if not (0 <= idx < len(self._pages)):
            raise ToolError(
                f"Tab {idx} does not exist. Open tabs: 0..{len(self._pages) - 1}."
            )
        page = self._pages.pop(idx)
        await page.close()
        if not self._pages:
            blank = await self._context.new_page()
            blank.set_default_timeout(config.timeout_ms())
            self._pages = [blank]
            self._active_index = 0
        else:
            self._active_index = min(idx, len(self._pages) - 1)
        return {
            "closed_tab": idx,
            "tabs_open": len(self._pages),
            "active_tab": self._active_index,
        }

    async def list_tabs(self) -> list[dict[str, Any]]:
        return [
            {
                "tab": i,
                "active": i == self._active_index,
                "url": page.url,
                "title": await page.title(),
            }
            for i, page in enumerate(self._pages)
        ]

    # -- reading -------------------------------------------------------------

    async def get_url(self) -> str:
        return self.active_page.url

    async def get_title(self) -> str:
        return await self.active_page.title()

    async def read_page(
        self, selector: str | None = None, max_chars: int = MAX_READ_CHARS
    ) -> str:
        page = self.active_page
        if selector:
            loc = page.locator(selector).first
            if not await loc.count():
                raise ToolError(f"No element matches selector {selector!r}.")
            text = await loc.inner_text()
        else:
            text = await page.locator("body").inner_text()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
        return text

    async def extract_links(self, max_links: int = MAX_LINKS) -> list[dict[str, str]]:
        page = self.active_page
        anchors = page.locator("a[href]")
        count = await anchors.count()
        links: list[dict[str, str]] = []
        seen: set[str] = set()
        for i in range(count):
            href = await anchors.nth(i).get_attribute("href")
            if not href:
                continue
            resolved = urljoin(page.url, href)
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                text = (await anchors.nth(i).inner_text()).strip() or href
            except PlaywrightError:
                text = href
            links.append({"text": text, "href": href, "url": resolved})
            if len(links) >= max_links:
                break
        return links

    async def inspect_page(self, max_items: int = MAX_INSPECT_ITEMS) -> str:
        """Return a numbered inventory of interactive elements on the page."""
        page = self.active_page
        self._inspect_items = {}
        lines: list[str] = []
        count = 0
        for role, selector in _GROUPS:
            if count >= max_items:
                break
            total = min(await page.locator(selector).count(), 200)
            for i in range(total):
                if count >= max_items:
                    break
                el = page.locator(selector).nth(i)
                try:
                    if not await el.is_visible():
                        continue
                    self._inspect_items[str(count + 1)] = el
                    lines.append(f"{count + 1}. {await self._describe(el, role)}")
                    count += 1
                except PlaywrightError:
                    continue
        header = f"URL: {page.url}\nTitle: {await page.title()}\n"
        if not lines:
            return header + "No interactive elements found on the page."
        footer = (
            f"\nTotal: {count} interactive elements (capped at {max_items}). "
            "Target them by number, 'role:name', visible text, or CSS selector."
        )
        return header + "\n".join(lines) + footer

    # -- interaction ---------------------------------------------------------

    async def click(self, target: str) -> str:
        loc = await self._resolve_target(target)
        try:
            await loc.click(timeout=config.timeout_ms())
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            raise ToolError(f"Could not click {target!r}: {exc}") from exc
        return f"Clicked {target!r}."

    async def type_text(self, target: str, text: str, submit: bool = False) -> str:
        loc = await self._resolve_target(target)
        try:
            await loc.fill(text, timeout=config.timeout_ms())
            if submit:
                await loc.press("Enter")
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            raise ToolError(f"Could not type into {target!r}: {exc}") from exc
        return f"Typed into {target!r}." + (" and pressed Enter." if submit else "")

    async def press_key(self, key: str) -> str:
        await self.active_page.keyboard.press(key)
        return f"Pressed {key}."

    async def clear_field(self, target: str) -> str:
        loc = await self._resolve_target(target)
        try:
            await loc.fill("")
        except PlaywrightError as exc:
            raise ToolError(f"Could not clear {target!r}: {exc}") from exc
        return f"Cleared {target!r}."

    async def hover(self, target: str) -> str:
        loc = await self._resolve_target(target)
        try:
            await loc.hover()
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            raise ToolError(f"Could not hover {target!r}: {exc}") from exc
        return f"Hovered over {target!r}."

    async def select_option(self, target: str, value: str) -> str:
        loc = await self._resolve_target(target)
        for how in ("value", "label"):
            try:
                await loc.select_option(**{how: value}, timeout=config.timeout_ms())
                return f"Selected {value!r} in {target!r}."
            except (PlaywrightTimeoutError, PlaywrightError):
                continue
        raise ToolError(f"Option {value!r} not found in {target!r}.")

    async def scroll(self, direction: str = "down", amount: int | None = None) -> str:
        if direction not in ("up", "down"):
            raise ToolError("Scroll direction must be 'up' or 'down'.")
        dy = abs(amount or 500)
        await self.active_page.evaluate(
            f"() => window.scrollBy(0, {dy if direction == 'down' else -dy})"
        )
        return f"Scrolled {direction}."

    async def wait_for(self, selector: str | None = None, wait_ms: int = 1000) -> str:
        page = self.active_page
        if selector:
            try:
                await page.locator(selector).first.wait_for(
                    state="visible", timeout=config.timeout_ms()
                )
            except PlaywrightTimeoutError as exc:
                raise ToolError(
                    f"Selector {selector!r} did not become visible within {config.timeout_ms()} ms."
                ) from exc
            return f"Waited for {selector!r} to be visible."
        await page.wait_for_timeout(max(0, wait_ms))
        return f"Waited {wait_ms} ms."

    # -- screenshots ---------------------------------------------------------

    async def screenshot(self, full_page: bool = False) -> str:
        page = self.active_page
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = _SCREENSHOT_DIR / f"jarvis-{time.strftime('%Y%m%d-%H%M%S')}.png"
        await page.screenshot(path=str(path), full_page=full_page)
        return str(path)

    # -- cookies / storage ---------------------------------------------------

    async def get_cookies(self) -> list[dict[str, str | None]]:
        if self._context is None:
            raise ToolError("The browser is not open. Open a URL first.")
        return [
            {
                "name": c["name"],
                "domain": c["domain"],
                "path": c["path"],
                "httpOnly": c["httpOnly"],
                "secure": c["secure"],
                "sameSite": c["sameSite"] or None,
            }
            for c in await self._context.cookies()
        ]

    async def clear_cookies(self) -> str:
        if self._context is None:
            raise ToolError("The browser is not open. Open a URL first.")
        count = len(await self._context.cookies())
        await self._context.clear_cookies()
        return f"Cleared {count} cookies."

    # -- uploads / downloads -------------------------------------------------

    async def upload_file(self, target: str, path: str) -> str:
        loc = await self._resolve_target(target)
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise ToolError(f"File not found: {path}")
        await loc.set_input_files(str(file_path))
        return f"Uploaded {file_path.name} to {target!r}."

    async def download(self, target: str | None = None, url: str | None = None) -> str:
        page = self.active_page
        _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if target:
                loc = await self._resolve_target(target)
                async with page.expect_download(timeout=config.timeout_ms()) as dl:
                    await loc.click(timeout=config.timeout_ms())
            elif url:
                self._check_url(url)
                async with page.expect_download(timeout=config.timeout_ms()) as dl:
                    await page.goto(url, wait_until="load", timeout=config.timeout_ms())
            else:
                raise ToolError("download_file needs a target element or a url.")
        except PlaywrightTimeoutError as exc:
            raise ToolError("No download was triggered.") from exc
        download: Download = await dl.value
        dest = _DOWNLOAD_DIR / download.suggested_filename
        await download.save_as(str(dest))
        logger.info("Downloaded %s -> %s", download.suggested_filename, dest)
        return str(dest)

    # -- javascript ----------------------------------------------------------

    async def run_javascript(self, code: str) -> str:
        if not config.allow_js():
            raise ToolError(
                "JavaScript execution is disabled. Set BROWSER_ALLOW_JS=1 to enable it."
            )
        try:
            result = await self.active_page.evaluate(code)
        except PlaywrightError as exc:
            raise ToolError(f"JavaScript error: {exc}") from exc
        if result is None:
            return "Executed; the code returned nothing."
        return str(result)
