import re
from bs4 import BeautifulSoup
from dateutil import parser
from typing import Optional

def clean_text(s: Optional[str], strip_html: bool = False) -> Optional[str]:
    if s is None: return None
    if strip_html:
        soup = BeautifulSoup(s, "lxml")
        s = soup.get_text(" ", strip=True)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_iso8601(dt: Optional[str]) -> Optional[str]:
    if not dt: return None
    try:
        return parser.parse(dt).isoformat()
    except Exception:
        return dt
