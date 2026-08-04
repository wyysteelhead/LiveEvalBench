"""Browser connection handler for Chrome DevTools Protocol (CDP)."""

import asyncio
from typing import Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page


class BrowserConnector:
    """Manages Playwright connection to Chromium via CDP.

    Handles browser lifecycle and provides page context for automation.
    """

    def __init__(self):
        """Initialize browser connector."""
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def connect(self, cdp_url: str, timeout: int = 30000) -> Page:
        """Connect to Chromium via CDP and create a page.

        Args:
            cdp_url: Chrome DevTools Protocol WebSocket URL
            timeout: Connection timeout in milliseconds

        Returns:
            Playwright Page instance

        Raises:
            RuntimeError: If connection fails
        """
        try:
            self.playwright = await async_playwright().start()

            print(f"Connecting to browser via CDP: {cdp_url}")

            deadline = asyncio.get_running_loop().time() + (timeout / 1000)
            last_error: Exception | None = None

            for attempt in range(1, 4):
                remaining_ms = max(1000, int((deadline - asyncio.get_running_loop().time()) * 1000))
                try:
                    self.browser = await self.playwright.chromium.connect_over_cdp(
                        cdp_url,
                        timeout=remaining_ms,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 3:
                        raise
                    await asyncio.sleep(0.5 * attempt)

            # Create browser context
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
                ignore_https_errors=True,
            )
            await self.context.add_init_script("""(() => {
              const fonts = document.fonts;
              if (!fonts) return;
              const origReady = fonts.ready;
              const timeout = new Promise(resolve => setTimeout(resolve, 30000));
              Object.defineProperty(fonts, 'ready', {
                get: () => Promise.race([origReady, timeout]),
                configurable: true,
              });
            })();
            """)

            # Create page
            self.page = await self.context.new_page()

            print("✓ Browser connected successfully")
            return self.page

        except Exception as e:
            await self.close()
            raise RuntimeError(f"Failed to connect to browser: {e}")

    async def close(self) -> None:
        """Close browser connection and cleanup resources."""
        if self.page:
            try:
                await self.page.close()
            except Exception as e:
                print(f"Warning: Error closing page: {e}")

        if self.context:
            try:
                await self.context.close()
            except Exception as e:
                print(f"Warning: Error closing context: {e}")

        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                print(f"Warning: Error closing browser: {e}")

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                print(f"Warning: Error stopping playwright: {e}")

        print("✓ Browser connection closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
