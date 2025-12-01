import re
from typing import Any, Dict, List, Tuple, Optional
from bs4 import BeautifulSoup

def to_text(css: str) -> str:
    return css if "::" in css else f"{css}::text"

def preview_value(html: str, selector: str, scope_css: Optional[str] = None,
                    join_all: bool = False, sep: str = " ") -> Dict[str, Any]:
    """Simple preview using BeautifulSoup for dev."""
    soup = BeautifulSoup(html or "", "lxml")
    root = soup.select_one(scope_css) if scope_css else soup
    if not root:
        return {"count": 0, "value": ""}

    attr = None
    css = selector
    if "::attr(" in selector:
        m = re.search(r"::attr\(([^)]+)\)", selector)
        if m:
            attr = m.group(1)
            css = selector.split("::attr(")[0]
    elif selector.endswith("::text"):
        css = selector[:-6]

    nodes = root.select(css) if css else []
    if not nodes:
        return {"count": 0, "value": ""}

    vals: List[str] = []
    for n in nodes:
        if attr:
            v = n.get(attr) or ""
        else:
            v = n.get_text(" ", strip=True)
        if v:
            vals.append(v.strip())

    if join_all:
        return {"count": len(vals), "value": (sep or " ").join(vals)}
    return {"count": len(vals), "value": (vals[0] if vals else "")}

def absolutize_url(url: Optional[str], base: str) -> Optional[str]:
    if not url:
        return None
    if url.startswith("http"):
        return url
    if base:
        try:
            from urllib.parse import urljoin
            return urljoin(base, url)
        except Exception:
            return url
    return url