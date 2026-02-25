"""Type stubs for playwright-stealth (no py.typed marker in upstream)."""

from playwright.async_api import BrowserContext, Page

class Stealth:
    async def apply_stealth_async(self, page_or_context: Page | BrowserContext) -> None: ...
