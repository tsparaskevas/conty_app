# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import re
import unicodedata

from bs4 import BeautifulSoup

def _fix_url_scheme(url: str) -> str:
    """
    Normalize URLs like 'https:/foo' -> 'https://foo'.

    Be VERY conservative: only fix if we see exactly 'http:/' or 'https:/'
    at the start, and not when the URL already has '://'.
    """
    if not isinstance(url, str):
        return url

    u = url.strip()
    # Already a normal URL? Leave it alone.
    if u.startswith("http://") or u.startswith("https://"):
        return u

    # Fix malformed single-slash schemes only
    if u.startswith("http:/"):
        # 'http:/' + 'rest' -> 'http://' + 'rest'
        return "http://" + u[len("http:/"):]
    if u.startswith("https:/"):
        # 'https:/' + 'rest' -> 'https://' + 'rest'
        return "https://" + u[len("https:/"):]
    return u

# ------------------------------------------------------------
# Date parsing helpers (Greek + English tokens) — article-friendly
# ------------------------------------------------------------

# 20/09/2025 • 00:00 •  (also accepts dash, comma, or vertical bar)
RE_DDMMYYYY_BULLET_HHMM = re.compile(
    r"""^\s*
        (\d{1,2})[./-](\d{1,2})[./-](\d{2,4})     # DD/MM/YYYY
        \s*[-\u2013\u2014\u00B7\u2022,|]\s*       # -, – (en dash), — (em dash), ·, comma, |
        (\d{1,2}):(\d{2})(?::(\d{2}))?            # HH:MM[:SS]
        \s*(?:[\u00B7\u2022|]\s*)?$                     # optional trailing · or |
    """,
    re.X,
)

# 19/09/2025 - 20:00   (also accepts en/em dash and optional seconds)
RE_DDMMYYYY_DASH_HHMM = re.compile(
    r"""^\s*
        (\d{1,2})[./-](\d{1,2})[./-](\d{2,4})
        \s*[-\u2013\u2014]\s*
        (\d{1,2}):(\d{2})(?::(\d{2}))?
        \s*$
    """,
    re.X,
)

# 07:27 17/09/2025 or 07:2717/09/2025
RE_HHMM_DDMMYYYY_COMPACT = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\s*$"
)

# 19.09.25 13:41  (space optional; year 2 or 4 digits)
RE_DDMMYYYY_HHMM_COMPACT = re.compile(
    r"^\s*(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})(?=\s*\d{1,2}:)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
)

# "19/09/2025 10:10"
RE_DDMMYYYY_HHMM = re.compile(
    r"^\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
)

# Weekday (optional), DD-Mon-YYYY HH:MM[:SS]   e.g. "Παρασκευή, 19-Σεπ-2025 12:26"
RE_WD_DD_MON_YYYY_HHMM = re.compile(
    r"""(?ix)
    ^\s*
    (?:(?P<wd>[\w.\u0370-\u03FF\u1F00-\u1FFF]+)\s*,\s*)?   # optional weekday + comma
    (?P<day>\d{1,2})
    [./-]
    (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)
    [./-]
    (?P<year>\d{2,4})
    \s+
    (?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?
    \s*$
    """
)

# "11:37 09/07" (assume current year)
RE_HHMM_DDMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*(\d{2})/(\d{2})\s*$")

# "28/09/2023", "28-09-23", "28.09.2023"
RE_DDMMYYYY = re.compile(r"^\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\s*$")

# "28/09" (assume current year)
RE_DDMM = re.compile(r"^\s*(\d{1,2})[./-](\d{1,2})\s*$")

# Day Mon [Year] [HH:MM[:SS]]  (supports Greek/English tokens, optional commas/dashes/bullets)
RE_DAY_MON_YEAR_TIME = re.compile(
    r"""(?ix)
    ^\s*
    (?P<day>\d{1,2})
    (?:\s*,\s*|\s+)
    (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)
    (?: (?:\s*,\s*|\s+) (?P<year>\d{2,4}) )?
    (?: (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*) (?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))? )?
    \s*$
    """
)

# Weekday + Day Mon Year Time
# e.g. "Κυριακή 23 Νοεμβρίου 2025, 14:58:30"
RE_WD_DAY_MON_YEAR_TIME = re.compile(
    r"""(?ix)
    ^\s*
    (?P<wd>[\w.\u0370-\u03FF\u1F00-\u1FFF]+)          # weekday (Greek or Latin)
    (?:\s*,\s*|\s+)                                   # space or comma
    (?P<day>\d{1,2})
    (?:\s*,\s*|\s+)
    (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)     # month (Greek/English)
    (?:\s*,\s*|\s+)
    (?P<year>\d{2,4})
    (?:\s*,\s*|\s+)
    (?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?  # time HH:MM[:SS]
    \s*$
    """
)

# 07:00 [sep] 1 Σεπτεμβρίου 2025 (time first, optional seconds/weekday, flexible separators)
RE_HHMM_DAY_MON_YEAR = re.compile(
    r"""(?ix)
    ^\s*
    (?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?
    (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*)      # <- accept comma/space/dash/bullet/bar
    (?P<day>\d{1,2})
    (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*)
    (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)
    (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*)(?P<year>\d{2,4})?
    \s*$
    """
)

RE_HHMM_WD_DAY_MON_YEAR = re.compile(
    r"""(?ix)
    ^\s*
    (?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?
    (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*)
    (?:(?P<wd>[\w.\u0370-\u03FF\u1F00-\u1FFF]+)(?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*))?
    (?P<day>\d{1,2})
    (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*)
    (?P<mon>[\w.\u0370-\u03FF\u1F00-\u1FFF]+)
    (?:\s*,\s*|\s+|\s*[-\u2013\u2014\u00B7,|]\s*)(?P<year>\d{2,4})?
    \s*$
    """
)

RE_MON_DAY_YEAR = re.compile(
    r"""(?ix)
    ^\s*
    (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)
    (?:\s*,\s*|\s+)
    (?P<day>\d{1,2})
    (?:\s*,\s*|\s+)
    (?P<year>\d{2,4})
    \s*$
    """
)

# MonDDYYYY (compact, no separators), e.g. "Σεπ172025", "Sep172025"
RE_MON_DDYYYY_COMPACT = re.compile(
    r"""(?ix)^\s*
        (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)\s*
        (?P<day>\d{1,2})\s*
        (?P<year>\d{2,4})
        \s*$"""
)

# MonDDYYYY HH:MM[:SS], e.g. "Σεπ172025 07:30"
RE_MON_DDYYYY_HHMM_COMPACT = re.compile(
    r"""(?ix)^\s*
        (?P<mon>[A-Za-z\u0370-\u03FF\u1F00-\u1FFF.]+)\s*
        (?P<day>\d{1,2})\s*
        (?P<year>\d{2,4})
        \s+(?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?
        \s*$"""
)

# English "x ago"
RE_EAGO = [
    re.compile(r"(?i)\b(\d+)\s*seconds?\s*ago\b"),
    re.compile(r"(?i)\b(\d+)\s*mins?\s*ago\b"),
    re.compile(r"(?i)\b(\d+)\s*hours?\s*ago\b"),
    re.compile(r"(?i)\b(\d+)\s*days?\s*ago\b"),
    re.compile(r"(?i)\b(\d+)\s*weeks?\s*ago\b"),
    re.compile(r"(?i)\b(\d+)\s*months?\s*ago\b"),
    re.compile(r"(?i)\b(\d+)\s*years?\s*ago\b"),
]

# Greek tokens via Unicode escapes (editor-proof)
G_PRIN = r"\u03C0\u03C1\u03B9\u03BD"  # "πριν"
G_APO  = r"\u03B1\u03C0\u03CC"        # "από"
G_SECS = r"(?:\u03B4\u03B5\u03C5\u03C4\u03B5\u03C1\u03CC\u03BB\u03B5\u03C0\u03C4\u03B1|\u03B4\u03B5\u03C5\u03C4\.?|sec|seconds?)"
G_MINS = r"(?:\u03BB\u03B5\u03C0\u03C4\u03AC|\u03BB\u03B5\u03C0\u03C4\u03CC|\u03BB\u03B5\u03C0\u03C4\.?|min|minutes?)"
G_HOUR = r"(?:\u03CE\u03C1\u03B5\u03C2|\u03CE\u03C1\u03B1|\u03C9\u03C1\u03B5\u03C2|hour|hours?)"
G_DAYS = r"(?:\u03B7\u03BC\u03AD\u03C1\u03B5\u03C2|\u03B7\u03BC\u03AD\u03C1\u03B1|\u03BC\u03AD\u03C1\u03B5\u03C2|\u03BC\u03AD\u03C1\u03B1|day|days?)"
G_WEE  = r"(?:\u03B5\u03B2\u03B4\u03BF\u03BC\u03AC\u03B4\u03B5\u03C2|\u03B5\u03B2\u03B4\u03BF\u03BC\u03AC\u03B4\u03B1|week|weeks?)"
G_MONS = r"(?:\u03BC\u03AE\u03BD\u03B5\u03C2|\u03BC\u03B7\u03BD\u03B5\u03C2|month|months?)"
G_YEAS = r"(?:\u03C7\u03C1\u03CC\u03BD\u03B9\u03B1|\u03AD\u03C4\u03B7|year|years?)"

RE_GAGO = [
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_SECS}\b"),
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_MINS}\b"),
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_HOUR}\b"),
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_DAYS}\b"),
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_WEE}\b"),
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_MONS}\b"),
    re.compile(fr"(?iu)\b{G_PRIN}(?:\s+{G_APO})?\s+(\d+)\s*{G_YEAS}\b"),
]

RE_GBARE = [
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_SECS}\s*$"),
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_MINS}\s*$"),
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_HOUR}\s*$"),
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_DAYS}\s*$"),
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_WEE}\s*$"),
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_MONS}\s*$"),
    re.compile(fr"(?iu)^\s*(\d+)\s*{G_YEAS}\s*$"),
]

_G_ACCENTS = str.maketrans({
    "ά":"α","έ":"ε","ί":"ι","ό":"ο","ύ":"υ","ή":"η","ώ":"ω",
    "ϊ":"ι","ϋ":"υ","ΐ":"ι","ΰ":"υ",
    "Ά":"Α","Έ":"Ε","Ί":"Ι","Ό":"Ο","Ύ":"Υ","Ή":"Η","Ώ":"Ω",
})

MONTHS: Dict[str, int] = {
    # Greek short
    "ιαν":1,"φεβ":2,"μαρ":3,"απρ":4,"μαι":5,"μαϊ":5,"ιουν":6,"ιουλ":7,"αυγ":8,"σεπ":9,"σεπτ":9,"οκτ":10,"νοε":11,"δεκ":12,
    # Greek long (genitive & nominative)
    "ιανουαριου":1,"ιανουαριος":1,"φεβρουαριου":2,"φεβρουαριος":2,"μαρτιου":3,"μαρτιος":3,"απριλιου":4,"απριλιος":4,
    "μαιου":5,"μαιος":5,"ιουνιου":6,"ιουνιος":6,"ιουλιου":7,"ιουλιος":7,"αυγουστου":8,"αυγουστος":8,
    "σεπτεμβριου":9,"σεπτεμβριος":9,"οκτωβριου":10,"οκτωβριος":10,"νοεμβριου":11,"νοεμβριος":11,"δεκεμβριου":12,"δεκεμβριος":12,
    # English
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
    "aug":8,"august":8,"sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12,
}

def _preclean(text: str) -> str:
    # If we accidentally got an HTML snippet (e.g. "<time ...>Κυριακή ...</time> /"),
    # strip tags and keep only the visible text.
    if "<" in text and ">" in text:
        try:
            soup = BeautifulSoup(text, "lxml")
            text = soup.get_text(" ", strip=True)
        except Exception:
            # Fallback: crude tag removal
            text = re.sub(r"<[^>]+>", " ", text)

    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"(?i)\bupd\s*[:\-]\s*", "", t)
    t = re.sub(r"(?i)\bupd\b\s*[:\-]?\s*", "", t)
    t = re.sub(r"(?i)\bupdated\s*[:\-]\s*", "", t)
    t = re.sub(r"(?i)\bupdate\s*[:\-]\s*", "", t)
    t = t.strip("()[]")
    t = t.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2009", " ")
    t = t.replace("•", "·")
    t = t.replace("،", ",").replace("，", ",")
    t = re.sub(r"\s+", " ", t).strip()
    # Matches: " / Τελευταία Ενημέρωση: 12:34" (seconds optional) at the end of the string
    t = re.sub(
        r"""(?ix)                                # ignore case, verbose
            \s*/\s*
            (?:Τελευταία\s+Ενημέρωση|Τελευταία\s+ ενημέρωση|Last\s+Update|Last\s+Updated|Updated)
            \s*:?\s*\d{1,2}:\d{2}(?::\d{2})?     # HH:MM[:SS]
            \s*$                                 # end of string
        """,
        "",
        t,
    )
    # Remove trailing slash if any leftover, e.g. "... 14:58:30 /"
    t = re.sub(r"\s*/\s*$", "", t)
    return t

def normalize_datetime_smart(
    text: Optional[str],
    site: Optional[str] = None,
    *,
    log_suggestions: bool = False,
    url_for_log: Optional[str] = None,
    which: str = "published",
) -> Optional[str]:
    """
    Backwards-compatible wrapper around normalize_datetime.

    - `site`, `log_suggestions`, `url_for_log` and `which` are accepted for
      compatibility but are ignored.
    - Returns whatever normalize_datetime(text) returns.
    """
    return normalize_datetime(text)

def _month_to_num(token: str) -> Optional[int]:
    if not token:
        return None
    t = token.strip().strip(".").lower().translate(_G_ACCENTS)
    t = t.replace("ϊ","ι").replace("ΐ","ι").replace("ϋ","υ").replace("ΰ","υ")
    return MONTHS.get(t)

def _apply_ago(now: datetime, text: str) -> Optional[str]:
    t = text.strip()
    # English
    for i, rex in enumerate(RE_EAGO):
        m = rex.search(t)
        if m:
            val = int(m.group(1))
            delta = [
                timedelta(seconds=val),
                timedelta(minutes=val),
                timedelta(hours=val),
                timedelta(days=val),
                timedelta(weeks=val),
                timedelta(days=val*30),
                timedelta(days=val*365),
            ][i]
            return (now - delta).strftime("%Y-%m-%dT%H:%M")
    # Greek ("πριν X ...")
    for i, rex in enumerate(RE_GAGO):
        m = rex.search(t)
        if m:
            val = int(m.group(1))
            delta = [
                timedelta(seconds=val),
                timedelta(minutes=val),
                timedelta(hours=val),
                timedelta(days=val),
                timedelta(weeks=val),
                timedelta(days=val*30),
                timedelta(days=val*365),
            ][i]
            return (now - delta).strftime("%Y-%m-%dT%H:%M")
    # Bare Greek unit without "πριν" (e.g. "3 ώρες")
    for i, rex in enumerate(RE_GBARE):
        m = rex.search(t)
        if m:
            val = int(m.group(1))
            delta = [
                timedelta(seconds=val),
                timedelta(minutes=val),
                timedelta(hours=val),
                timedelta(days=val),
                timedelta(weeks=val),
                timedelta(days=val*30),
                timedelta(days=val*365),
            ][i]
            return (now - delta).strftime("%Y-%m-%dT%H:%M")
    return None

def normalize_datetime(text: Optional[str]) -> Optional[str]:
    """Normalize various publish/update datetime strings into ISO-like 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'."""
    if not text:
        return None
    t = _preclean(text)
    now = datetime.now()

    # "x ago" forms
    ago = _apply_ago(now, t)
    if ago:
        return ago

    # time-first with dot date: "16:36, 26.07.2024" ---
    m = re.match(
        r"""^\s*
            (\d{1,2})              # hour
            :
            (\d{2})                # minute
            (?::(\d{2}))?          # optional seconds
            \s*,\s*
            (\d{1,2})              # day
            \.
            (\d{1,2})              # month
            \.
            (\d{4})                # year
            \s*$
        """,
        t,
        re.VERBOSE,
    )
    if m:
        hh, mi, ss, dd, mm, yyyy = m.groups()
        hh = hh.zfill(2)
        mi = mi.zfill(2)
        dd = dd.zfill(2)
        mm = mm.zfill(2)
        if ss:
            ss = ss.zfill(2)
            return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}"
        else:
            return f"{yyyy}-{mm}-{dd}T{hh}:{mi}"

    # strict DD.MM.YYYY,DD.MM.YYYY HH:MM with dots
    m = re.match(r"^\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*,?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$", t)
    if m:
        dd, MM, yy, hh, mi, ss = m.groups()
        return f"{yy}-{MM.zfill(2)}-{dd.zfill(2)}T{hh.zfill(2)}:{mi}"

    m = RE_MON_DDYYYY_HHMM_COMPACT.match(t)
    if m:
        mon_tok = m.group("mon"); dd = int(m.group("day")); yy = int(m.group("year"))
        hh = int(m.group("hh")); mi = int(m.group("mm")); ss = int(m.group("ss")) if m.group("ss") else 0
        mon = _month_to_num(mon_tok)
        if mon:
            yy = 2000 + yy if yy < 100 else yy
            try:
                return datetime(yy, mon, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
            except ValueError:
                return None

    m = RE_MON_DDYYYY_COMPACT.match(t)
    if m:
        mon_tok = m.group("mon"); dd = int(m.group("day")); yy = int(m.group("year"))
        mon = _month_to_num(mon_tok)
        if mon:
            yy = 2000 + yy if yy < 100 else yy
            try:
                return datetime(yy, mon, dd).strftime("%Y-%m-%d")
            except ValueError:
                return None
    
    m = RE_WD_DD_MON_YYYY_HHMM.match(t)
    if m:
        dd = int(m.group("day"))
        mon_tok = m.group("mon")
        yy_s = m.group("year")
        hh = int(m.group("hh")); mi = int(m.group("mm"))
        ss = int(m.group("ss")) if m.group("ss") else 0
        mon = _month_to_num(mon_tok)
        if mon:
            yy = int(yy_s); yy = 2000 + yy if yy < 100 else yy
            try:
                return datetime(yy, mon, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
            except ValueError:
                return None

    m = RE_DDMMYYYY_BULLET_HHMM.match(t) or RE_DDMMYYYY_DASH_HHMM.match(t)
    if m:
        dd_s, MM_s, yy_s, hh_s, mm_s, ss_s = m.groups()
        dd, MM = int(dd_s), int(MM_s)
        yy = int(yy_s); yy = 2000 + yy if yy < 100 else yy
        hh, mi = int(hh_s), int(mm_s)
        ss = int(ss_s) if ss_s else 0
        try:
            return datetime(yy, MM, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            return None

    m = RE_DDMMYYYY_HHMM_COMPACT.match(t)
    if m:
        dd_s, MM_s, yy_s, hh_s, mm_s, ss_s = m.groups()
        dd, MM = int(dd_s), int(MM_s)
        yy = int(yy_s); yy = 2000 + yy if yy < 100 else yy
        hh, mi = int(hh_s), int(mm_s)
        ss = int(ss_s) if ss_s else 0
        try:
            return datetime(yy, MM, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            return None

    m = RE_HHMM_DDMMYYYY_COMPACT.match(t)
    if m:
        hh_s, mm_s, dd_s, MM_s, yy_s = m.groups()
        hh, mm = int(hh_s), int(mm_s)
        dd, MM = int(dd_s), int(MM_s)
        yy = int(yy_s); yy = 2000 + yy if yy < 100 else yy
        try:
            return datetime(yy, MM, dd, hh, mm).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            return None

    m = RE_HHMM_WD_DAY_MON_YEAR.match(t) or RE_HHMM_DAY_MON_YEAR.match(t)
    if m:
        dd = int(m.group("day")); mon_tok = m.group("mon"); year_s = m.group("year")
        hh = int(m.group("hh")); mi = int(m.group("mm")); ss = m.group("ss")
        mon = _month_to_num(mon_tok)
        if mon:
            yy = int(year_s) if year_s else now.year
            yy = 2000 + yy if yy < 100 else yy
            try:
                return datetime(yy, mon, dd, hh, mi, int(ss) if ss else 0).strftime("%Y-%m-%dT%H:%M")
            except ValueError:
                return None

    m = RE_DDMMYYYY_HHMM.match(t)
    if m:
        dd, MM, yy, hh, mi, ss = m.groups()
        dd, MM = int(dd), int(MM)
        yy = int(yy); yy = 2000 + yy if yy < 100 else yy
        hh = int(hh); mi = int(mi); ss = int(ss) if ss else 0
        try:
            return datetime(yy, MM, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            return None

    m = RE_HHMM_DDMM.match(t)
    if m:
        hh, mm, dd, MM = map(int, m.groups())
        try:
            return datetime(now.year, MM, dd, hh, mm).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            return None

    m = RE_DDMMYYYY.match(t)
    if m:
        dd, MM, yy = m.groups()
        dd, MM = int(dd), int(MM)
        yy = int(yy); yy = 2000 + yy if yy < 100 else yy
        try:
            return datetime(yy, MM, dd).strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = RE_DDMM.match(t)
    if m:
        dd, MM = map(int, m.groups())
        try:
            return datetime(now.year, MM, dd).strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = RE_MON_DAY_YEAR.match(t)
    if m:
        ...
        if mon:
            try:
                return datetime(yy, mon, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

    # "Κυριακή 23 Νοεμβρίου 2025, 14:58:30"
    m = RE_WD_DAY_MON_YEAR_TIME.match(t)
    if m:
        dd = int(m.group("day"))
        mon_tok = m.group("mon")
        year_s = m.group("year")
        hh = int(m.group("hh"))
        mi = int(m.group("mm"))
        ss = int(m.group("ss")) if m.group("ss") else 0
        mon = _month_to_num(mon_tok)
        if mon:
            yy = int(year_s) if year_s else now.year
            yy = 2000 + yy if yy < 100 else yy
            try:
                return datetime(yy, mon, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
            except ValueError:
                return None

    m = RE_DAY_MON_YEAR_TIME.match(t)
    if m:
        dd = int(m.group("day")); mon_tok = m.group("mon"); year_s = m.group("year")
        hh = m.group("hh"); mm = m.group("mm"); ss = m.group("ss")
        mon = _month_to_num(mon_tok)
        if mon:
            yy = int(year_s) if year_s else now.year
            yy = 2000 + yy if yy < 100 else yy
            try:
                if hh and mm:
                    return datetime(yy, mon, dd, int(hh), int(mm), int(ss) if ss else 0).strftime("%Y-%m-%dT%H:%M")
                return datetime(yy, mon, dd).strftime("%Y-%m-%d")
            except ValueError:
                return None

    return t  # fallback to original (cleaned) text

# --------- Text tidy helpers ---------
_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")  # zero-width chars

def tidy_text(s: Optional[str]) -> Optional[str]:
    """
    Normalize whitespace AND strip simple HTML markup.

    This is applied to title / subtitle / lead / section / author / tags
    in postprocess_article_dict, so changing it affects both Builder
    and Runner in a consistent way.
    """
    if s is None:
        return None

    t = str(s)

    # If it looks like HTML, strip tags first
    if "<" in t and ">" in t:
        try:
            soup = BeautifulSoup(t, "lxml")
            # get visible text, joining with spaces
            t = soup.get_text(" ", strip=True)
        except Exception:
            # if BeautifulSoup fails for some reason, keep original
            pass

    # Normalize spaces and punctuation spacing
    t = t.replace("\xa0", " ")
    t = _ZW_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t

# --------- HTML meta fallback for datetimes ----------
META_DT_KEYS = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "article:published_time"}),
    ("meta", {"property": "og:published_time"}),
    ("meta", {"property": "article:modified_time"}),
    ("meta", {"name": "article:modified_time"}),
    ("meta", {"property": "og:updated_time"}),
    ("time", {"datetime": True}),  # any <time datetime="...">
]

def _get_attr(el) -> Optional[str]:
    # prefer content / datetime / value / text
    for k in ("content", "datetime", "value"):
        v = el.attrs.get(k)
        if v:
            return v
    return el.get_text(strip=True) or None

def sniff_datetimes_from_html(html: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (published, updated) strings if found in meta/markup."""
    if not html:
        return (None, None)
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return (None, None)

    published_raw = None
    updated_raw = None
    for tag, attrs in META_DT_KEYS:
        if tag == "time" and attrs.get("datetime") is True:
            for t in soup.find_all("time"):
                v = _get_attr(t)
                if v and not published_raw:
                    published_raw = v
        else:
            el = soup.find(tag, attrs=attrs)
            if el and not published_raw:
                published_raw = _get_attr(el)

    # updated: search again with 'modified/updated'
    for prop in ("article:modified_time", "og:updated_time", "modified_time", "updated_time"):
        el = soup.find("meta", {"property": prop}) or soup.find("meta", {"name": prop})
        if el and not updated_raw:
            updated_raw = _get_attr(el)

    return (published_raw, updated_raw)

# Featured image: get the first url from srcset
def first_srcset_url(value: str | None) -> str:
    if not value:
        return ""
    # value like: "url1 400w, url2 800w"
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        # take the first token before any size (400w, 2x, etc)
        return part.split()[0]
    return ""

# --------- Public API for conty ---------
def postprocess_article_dict(
    d: Dict,
    html_for_fallback: Optional[str] = None,
    *,
    site_hint: Optional[str] = None,
    log_dt_suggestions: bool = False,
    url_for_log: Optional[str] = None,
) -> Dict:
    """
    Mutate/return a cleaned article dict:
      - normalize published_time / updated_time strings
      - tidy title/subtitle/lead/section/author/tags
      - if dates missing, sniff from HTML meta tags
    """
    out = dict(d)

    # --- small helper: strip HTML tags from candidate datetime strings ---
    def _strip_html_for_dt(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        s = str(value)
        if "<" in s and ">" in s:
            try:
                soup = BeautifulSoup(s, "lxml")
                s = soup.get_text(" ", strip=True)
            except Exception:
                # if BeautifulSoup fails, keep original
                pass
        s = s.strip()
        return s or None

    # Gather raw date strings (could be from YAML selectors)
    pub_raw = out.get("published_time") or out.get("published") or out.get("date") or None
    upd_raw = out.get("updated_time") or out.get("modified") or None

    # ----------------- published / updated datetimes -----------------
    pub_raw = out.get("published_time") or out.get("published") or out.get("date") or None
    upd_raw = out.get("updated_time") or out.get("modified") or None

    # Fallback to HTML meta if missing
    if (not pub_raw or not upd_raw) and html_for_fallback:
        s_pub, s_upd = sniff_datetimes_from_html(html_for_fallback)
        pub_raw = pub_raw or s_pub
        upd_raw = upd_raw or s_upd

    # Strip HTML from the raw strings before we do anything else
    pub_raw_clean = _strip_html_for_dt(pub_raw)
    upd_raw_clean = _strip_html_for_dt(upd_raw)

    # Keep the cleaned raw strings for downstream (CSV / debugging)
    out["published_raw"] = pub_raw_clean or ""
    out["updated_raw"] = upd_raw_clean or ""

    # Normalize to ISO-like strings
    if pub_raw_clean:
        out["published_time"] = normalize_datetime_smart(
            pub_raw_clean,
            site_hint,
            log_suggestions=log_dt_suggestions,
            url_for_log=url_for_log,
            which="published",
        )
    else:
        out["published_time"] = None

    if upd_raw_clean:
        out["updated_time"] = normalize_datetime_smart(
            upd_raw_clean,
            site_hint,
            log_suggestions=log_dt_suggestions,
            url_for_log=url_for_log,
            which="updated",
        )
    else:
        out["updated_time"] = None

    # Tidy a few text fields
    for k in ("title", "subtitle", "lead", "section", "author", "tags"):
        if k in out and out[k] is not None:
            out[k] = tidy_text(out[k])

    # If tags came as list, join to string (comma)
    if isinstance(out.get("tags"), list):
        out["tags"] = ", ".join([tidy_text(x) or "" for x in out["tags"] if x])

    # Normalize featured image:
    #  - srcset → first URL
    #  - <img ...> snippet → extract src / srcset
    #  - fix scheme issues like "https:/..."
    for key in ("main_image", "image"):
        val = out.get(key)
        if not isinstance(val, str) or not val.strip():
            continue

        v = val.strip()

        # Case 1: srcset-like string ("url1 400w, url2 800w")
        if "," in v and " " in v and "<" not in v:
            v = first_srcset_url(v)

        # Case 2: full or partial <img> HTML snippet
        elif "<img" in v:
            try:
                soup = BeautifulSoup(v, "lxml")
                img = soup.find("img")
                if img:
                    src = img.get("src") or first_srcset_url(img.get("srcset"))
                    if src:
                        v = src.strip()
            except Exception:
                # If BeautifulSoup fails, fall through and keep original val
                pass

        # In all cases, normalize scheme (e.g. https:/ → https://)
        v = _fix_url_scheme(v)

        out[key] = v

    return out

def postprocess_article_obj(article):
    """
    Accepts an Article dataclass or dict; returns the same type with fields normalized.
    """
    if is_dataclass(article):
        data = asdict(article)
        data = postprocess_article_dict(
            data,
            html_for_fallback=data.get("container_html") or data.get("body_html"),
        )
        for k, v in data.items():
            setattr(article, k, v)
        return article
    elif isinstance(article, dict):
        return postprocess_article_dict(
            article,
            html_for_fallback=article.get("container_html") or article.get("body_html"),
        )
    return article

