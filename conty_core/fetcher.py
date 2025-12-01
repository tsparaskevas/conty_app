from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import time

import requests

# --- Optional selenium imports are inside class to avoid import errors when unused ---

DEFAULT_UA = "Mozilla/5.0 (compatible; conty/1.0; +https://example.com/bot)"

# --- Shared Selenium driver (one per Streamlit session) ---
_shared_driver = None

def get_shared_driver(headless: bool = True, user_agent: str = DEFAULT_UA, page_load_strategy: str = "eager"):
    """
    Create or return a shared Selenium Chrome driver.
    """
    global _shared_driver
    if _shared_driver is not None:
        return _shared_driver

    # lazy imports
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    if headless:
        try:
            opts.add_argument("--headless=new")
        except Exception:
            opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--user-agent={user_agent}")
    try:
        opts.page_load_strategy = page_load_strategy  # "eager" | "normal" | "none"
    except Exception:
        pass

    service = Service(ChromeDriverManager().install())
    _shared_driver = webdriver.Chrome(service=service, options=opts)
    return _shared_driver

def close_shared_driver():
    global _shared_driver
    try:
        if _shared_driver is not None:
            _shared_driver.quit()
    except Exception:
        pass
    _shared_driver = None


@dataclass
class FetchResult:
    final_url: str
    html: str
    engine: str  # "Requests" or "Selenium"

class RequestsFetcher:
    def __init__(self, timeout: float = 25.0, user_agent: str = DEFAULT_UA):
        self.timeout = timeout
        self.user_agent = user_agent

    def get(self, url: str, headers: Optional[dict] = None) -> Tuple[str, str]:
        hdrs = {"User-Agent": self.user_agent}
        if headers:
            hdrs.update(headers)
        r = requests.get(url, timeout=self.timeout, headers=hdrs)
        r.raise_for_status()
        return r.url, r.text

class SeleniumFetcher:
    def __init__(self, headless: bool = True, page_load_strategy: str = "eager", wait_timeout: float = 20.0, user_agent: str = DEFAULT_UA, driver=None):
        self.headless = headless
        self.page_load_strategy = page_load_strategy
        self.wait_timeout = wait_timeout
        self.user_agent = user_agent
        self._driver = driver  # may be injected (shared)

        # Only create a new driver if none was injected
        if self._driver is None:
            # lazy imports (so module can be imported without selenium installed)
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            opts = Options()
            # headless flag
            if self.headless:
                try:
                    opts.add_argument("--headless=new")
                except Exception:
                    opts.add_argument("--headless")

            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument(f"--user-agent={self.user_agent}")

            # page load strategy (supported on Selenium 4)
            try:
                opts.page_load_strategy = self.page_load_strategy  # "eager" | "normal" | "none"
            except Exception:
                pass

            # Selenium 4 style: use Service(...)
            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=opts)

    def _click_many_xpaths(self, driver, xpaths: Iterable[str]) -> bool:
        """
        Try to click all given XPaths.

        Returns True if at least one element was clicked (either in the
        top-level document or inside any iframe), False otherwise.
        """
        from selenium.webdriver.common.by import By

        clicked = False

        # 1) Try in top-level document
        for xp in xpaths or []:
            try:
                el = driver.find_element(By.XPATH, xp)
                if el and el.is_displayed():
                    el.click()
                    time.sleep(0.3)
                    clicked = True
            except Exception:
                continue

        if clicked:
            return True

        # 2) Try inside iframes (e.g. Cookiebot dialog)
        try:
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
        except Exception:
            iframes = []

        for frame in iframes:
            try:
                driver.switch_to.frame(frame)
                for xp in xpaths or []:
                    try:
                        el = driver.find_element(By.XPATH, xp)
                        if el and el.is_displayed():
                            el.click()
                            time.sleep(0.3)
                            clicked = True
                            break
                    except Exception:
                        continue
            except Exception:
                # ignore broken iframe
                pass
            finally:
                # ALWAYS go back to top-level
                try:
                    driver.switch_to.default_content()
                except Exception:
                    pass

            if clicked:
                break

        return clicked

    def get(
        self,
        url: str,
        wait_for_css: Optional[str] = None,
        consent_click_xpaths: Optional[Iterable[str]] = None,
        headers: Optional[dict] = None,   # kept for parity
    ) -> Tuple[str, str]:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        d = self._driver
        d.get(url)

        # try consent clicks first (give overlays like Cookiebot time to appear)
        if consent_click_xpaths:
            try:
                WebDriverWait(d, min(self.wait_timeout, 10)).until(
                    lambda drv: self._click_many_xpaths(drv, consent_click_xpaths)
                )
            except Exception:
                # best-effort: one last attempt without waiting
                try:
                    self._click_many_xpaths(d, consent_click_xpaths)
                except Exception:
                    pass

        # wait for a CSS (first non-empty if multiple separated by comma)
        sel = None
        if wait_for_css:
            for candidate in [s.strip() for s in wait_for_css.split(",") if s.strip()]:
                sel = candidate
                break

        if sel:
            try:
                WebDriverWait(d, self.wait_timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
            except Exception:
                # proceed anyway; some pages are render-complete without this matching
                pass

        html = d.page_source or ""
        final_url = d.current_url or url
        return final_url, html

    def close(self):
        # Do not close if this is the shared driver
        from selenium.webdriver.remote.webdriver import WebDriver as _WD
        if self._driver is None:
            return
        # if you want: keep a flag when injected; simplest: never close here
        pass

def fetch_with_fallback(
    url: str,
    *,
    container_css: str = "",
    item_css: str = "",
    force_js: bool = False,
    wait_timeout: float = 20.0,
    consent_click_xpaths: Optional[Iterable[str]] = None,
) -> FetchResult:
    """
    Try Requests first. If force_js is True OR the container/item CSS doesn't appear in the Requests HTML,
    fall back to Selenium and wait for the item/container CSS.
    """
    # 1) Requests
    req = RequestsFetcher()
    try:
        final_url, html = req.get(url, headers={})
        if not force_js:
            # quick heuristic: if we have a selector, ensure it's there
            if item_css.strip() or container_css.strip():
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "lxml")
                css = (item_css.strip() or container_css.strip())
                if css and soup.select(css):
                    return FetchResult(final_url, html, "Requests")
            else:
                # nothing to validate; accept
                return FetchResult(final_url, html, "Requests")
    except Exception:
        pass

    # 2) Selenium (shared driver)
    sel = SeleniumFetcher(
        headless=True,
        page_load_strategy="eager",
        wait_timeout=wait_timeout,
        user_agent=DEFAULT_UA,
        driver=get_shared_driver(headless=True, user_agent=DEFAULT_UA, page_load_strategy="eager"),
    )
    try:
        wait_for = item_css.strip() or container_css.strip() or None
        final_url, html = sel.get(
            url, wait_for_css=wait_for, consent_click_xpaths=consent_click_xpaths, headers={}
        )
        return FetchResult(final_url, html, "Selenium")
    finally:
        # DON'T close the shared driver here
        pass
