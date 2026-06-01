"""
Type stub for ``playwright-stealth``.

playwright-stealth does not ship type stubs.  This minimal stub covers
the public API used by the session manager: ``Stealth().apply_stealth_async(context)``.
"""

from playwright.async_api import BrowserContext

class Stealth:
    async def apply_stealth_async(self, context: BrowserContext) -> None: ...
