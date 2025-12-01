import re
from dataclasses import asdict
from datetime import datetime, timezone
from selectolax.parser import HTMLParser
from urllib.parse import urljoin

from conty_core.models import Article  # absolute import, avoids relative quirks
from conty_core.normalize import clean_text, to_iso8601
from .postprocess import postprocess_article_obj

# ------------------------- low-level selectors -------------------------

# grab the first URL inside any url(...) in a style string, regardless of the CSS property name
STYLE_URL_RE = re.compile(r"url\(\s*['\"]?([^'\"\)]+)", re.I)

def _sel_one(doc: HTMLParser, sel: str):
    attr = None
    text_mode = False
    if "::attr(" in sel:
        base, rest = sel.split("::attr(", 1)
        sel, attr = base, rest.rstrip(")")
    if sel.endswith("::text"):
        sel = sel.replace("::text", "")
        text_mode = True
    node = doc.css_first(sel)
    if not node:
        return None
    if attr:
        a = (attr or "").strip().lower()
        if a == "style_url":
            style = node.attributes.get("style") or ""
            m = STYLE_URL_RE.search(style)
            return m.group(1) if m else None
        # normal attribute
        return node.attributes.get(attr)
    return node.text(strip=True) if text_mode else node.html

def _sel_join(doc, sel: str):
    """Join inner HTML of all nodes matching `sel` (legacy body_html)."""
    nodes = doc.css(sel)
    if not nodes:
        return None
    return "\n".join(n.html for n in nodes)

def _sel_all_texts(doc, sel: str):
    """Return list of texts/attrs for all matches: supports ::text and ::attr(name)."""

    if "::attr(" in sel:
        base, rest = sel.split("::attr(", 1)
        css, attr = base, rest.rstrip(")")
        a = (attr or "").strip().lower()
        out = []
        for n in doc.css(css):
            if not n:
                continue
            if a == "style_url":
                style = n.attributes.get("style") or ""
                m = STYLE_URL_RE.search(style)
                if m:
                    out.append(m.group(1))
            else:
                v = n.attributes.get(attr)
                if v:
                    out.append(v)
        return out

    css = sel.replace("::text", "")
    return [n.text(strip=True) for n in doc.css(css) if n and n.text(strip=True)]

def _sel_node_html(doc, sel: str):
    """Return inner HTML of the first node matching `sel`."""
    n = doc.css_first(sel)
    return n.html if n else None

def _make_scoped_doc(doc: HTMLParser, mc_css: str) -> HTMLParser | None:
    """Return a document limited to the main container's inner HTML."""
    if not mc_css or not mc_css.strip():
        return None
    node = doc.css_first(mc_css.strip())
    if not node or not node.html:
        return None
    return HTMLParser(node.html)

def _apply_extract(doc: HTMLParser, rules: dict) -> dict:
    out = {}
    for field, rule in (rules or {}).items():
        if isinstance(rule, str):
            out[field] = _sel_one(doc, rule)

        elif isinstance(rule, dict) and "any" in rule:
            for opt in rule["any"]:
                val = _sel_one(doc, opt) if isinstance(opt, str) else None
                if val:
                    out[field] = val
                    break

        elif isinstance(rule, dict) and "join" in rule:
            out[field] = _sel_join(doc, rule["join"])

        elif isinstance(rule, dict) and "join_texts" in rule:
            cfg = rule["join_texts"]
            sel = cfg.get("selector", "")
            sep = cfg.get("sep", ", ")
            arr = _sel_all_texts(doc, sel) or []
            out[field] = sep.join([v for v in arr if v])

        elif isinstance(rule, dict) and "node" in rule:
            out[field] = _sel_node_html(doc, rule["node"])
    return out

# ------------------------- ordered body composer -------------------------

_BLOCK_TAGS = {"p", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "blockquote"}

def _has_ancestor_block(n):
    p = n.parent
    while p is not None:
        if p.tag in _BLOCK_TAGS:
            return True
        p = p.parent
    return False

def _container_text_with_exclusions_for_extract(root_html: str,
                                                container_css: str,
                                                exclude_selectors: list[str]) -> str:
    """
    Given a small HTML fragment (root_html), find container_css inside it,
    remove all subtrees matched by exclude_selectors, then compose ordered text
    in reading order (paragraphs, headings, lists, quotes).
    """
    if not root_html or not container_css.strip():
        return ""
    doc = HTMLParser(root_html)
    root = doc.css_first(container_css.strip())
    if not root:
        return ""

    # collect excluded nodes
    excluded = set()
    for sel in exclude_selectors or []:
        sel = (sel or "").strip()
        if not sel:
            continue
        try:
            for n in root.css(sel):
                excluded.add(n)
        except Exception:
            continue

    def _excluded(n):
        p = n
        while p is not None:
            if p in excluded:
                return True
            p = p.parent
        return False

    lines = []
    for n in root.traverse():
        if n.tag not in _BLOCK_TAGS or _has_ancestor_block(n):
            continue
        if _excluded(n):
            continue

        if n.tag == "p":
            text = n.text(separator=" ", strip=True)
            if text:
                lines.append(text)

        elif n.tag in {"h2", "h3", "h4", "h5", "h6"}:
            text = n.text(separator=" ", strip=True)
            if text:
                lines.append(text)

        elif n.tag in {"ul", "ol"}:
            items = [li.text(separator=" ", strip=True) for li in n.css("li")]
            items = [x for x in items if x]
            if items:
                if n.tag == "ul":
                    lines.extend([f"- {it}" for it in items])
                else:
                    lines.extend([f"{i+1}. {it}" for i, it in enumerate(items)])

        elif n.tag == "blockquote":
            qt = n.text(separator=" ", strip=True)
            if qt:
                lines.append(qt)

    return "\n\n".join(lines)

def _apply_boilerplate(text: str, regexes: list[str]) -> str:
    """Remove lines that match any of the given regexes; collapse extra blank lines."""
    if not text:
        return text
    patterns = []
    for r in regexes or []:
        try:
            patterns.append(re.compile(r))
        except re.error:
            # ignore broken patterns
            continue
    out = []
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

# ------------------------- acceptance helpers -------------------------

def _min_chars_from_template(t: dict) -> int | None:
    accept = (t.get("extract") or {}).get("accept") or {}
    val = accept.get("min_body_chars")
    try:
        return int(val) if val is not None else None
    except Exception:
        return None

# ------------------------- main entry -------------------------

def extract_article(
    url: str,
    html: str,
    site_cfg: dict,
    *,
    forced_template: str | None = None,
    lenient: bool = False
) -> Article:
    """
    Extract an article using the site's templates.
    - forced_template: if given, that template is tried first and skips url_regex/dom_probe gating.
    - lenient: if True, accept lower-score results (for builder preview).
    """
    doc = HTMLParser(html)
    # Precompute body_config main_container for scoping
    site_rules = site_cfg.get("templates", []) # just a local alias
    candidates = sorted(site_cfg.get("templates", []), key=lambda t: t.get("priority", 100))

    # Put the forced template first if requested
    if forced_template:
        candidates = [t for t in candidates if t.get("name") == forced_template] + \
                     [t for t in candidates if t.get("name") != forced_template]

    last_error_reason = "no_match"
    last_template_tried = None

    for t in candidates:
        t_name = t.get("name")
        last_template_tried = t_name

        # If not the forced template, enforce url_regex + dom_probe as usual
        if not (forced_template and t_name == forced_template):
            ur = t.get("url_regex")
            if ur:
                try:
                    if not re.search(ur, url):
                        continue
                except re.error:
                    # broken regex → skip template
                    continue
            probes = t.get("dom_probe", [])
            if probes and not all(doc.css_first(p) for p in probes):
                continue

        # Apply extraction rules for non-body fields
        rules = t.get("extract", {}) or {}

        # Build a scoped doc using the template's body_config.main_container_css
        body_cfg = rules.get("body_config") or {}
        mc_css = (body_cfg.get("main_container_css") or "").strip()
        scoped_doc = _make_scoped_doc(doc, mc_css)

        # Extract twice: first within the main container, then globally as fallback
        res_scoped = _apply_extract(scoped_doc, rules) if scoped_doc else {}
        res_global = _apply_extract(doc, rules)

        # Merge: prefer scoped values; fall back to global if missing/empty
        GLOBAL_ONLY = {"breadcrumbs", "canonical_url"}  
        res = {}
        for k in set(res_scoped) | set(res_global):
            if k in GLOBAL_ONLY:
                res[k] = res_global.get(k)
            else:
                v_sc = res_scoped.get(k)
                v_gl = res_global.get(k)
                res[k] = v_sc if (v_sc not in (None, "", [])) else v_gl

        from urllib.parse import urljoin, urlparse

        # Ensure title exists via global fallbacks if rule didn't catch it
        if not (res.get("title") or "").strip():
            res["title"] = (
                _sel_one(doc, "meta[property='og:title']::attr(content)")
                or _sel_one(doc, "meta[name='og:title']::attr(content)")
                or _sel_one(doc, "head title::text")
            )

        def _abs_url(maybe_url: str | None, base: str) -> str | None:
            if not maybe_url:
                return None
            u = maybe_url.strip()
            # protocol-relative //example.com/…
            if u.startswith("//"):
                return "https:" + u
            # already absolute
            if urlparse(u).scheme in ("http", "https"):
                return u
            # relative → make absolute
            return urljoin(base, u)

        # Ensure main_image is filled: try inline style background first, then head meta
        if not res.get("main_image"):
            res["main_image"] = (
                _sel_one(doc, "[style*='url(']::attr(style_url)")
                or _sel_one(doc, "meta[property='og:image']::attr(content)")
                or _sel_one(doc, "meta[name='og:image']::attr(content)")
                or _sel_one(doc, "meta[property='twitter:image']::attr(content)")
                or _sel_one(doc, "meta[name='twitter:image']::attr(content)")
            )

        # ---------- New body_config + container_text path ----------
        container_html = None
        ordered_text = None

        ct = rules.get("container_text") or {}
        excludes = ct.get("exclude") or []
        boiler   = ct.get("boilerplate_regex") or []

        body_cfg = rules.get("body_config") or {}
        mc_css = (body_cfg.get("main_container_css") or "").strip()
        bc_css = (body_cfg.get("body_container_css") or "").strip()

        if mc_css and bc_css:
            mc_node = doc.css_first(mc_css)
            if mc_node:
                inner = HTMLParser(mc_node.html or "")
                bnode = inner.css_first(bc_css)
                if bnode:
                    container_html = bnode.html or ""
                    # Build ordered text like the builder
                    ordered_text = _container_text_with_exclusions_for_extract(
                        "<div>" + container_html + "</div>", "div", excludes
                    )
                    if ordered_text:
                        ordered_text = _apply_boilerplate(ordered_text, boiler)

#        body_cfg = rules.get("body_config") or {}
#        mc_css = (body_cfg.get("main_container_css") or "").strip()
#        bc_css = (body_cfg.get("body_container_css") or "").strip()
#
#        # Try to build ordered_text from body_config.
#        # 1) Prefer nested (main_container -> body_container)
#        # 2) If that fails, try body_container on the whole document.
#        if bc_css:
#            candidate_html = None
#
#            # a) nested path: main container → body container
#            if mc_css:
#                mc_node = doc.css_first(mc_css)
#                if mc_node and mc_node.html:
#                    inner = HTMLParser(mc_node.html)
#                    bnode = inner.css_first(bc_css)
#                    if bnode and bnode.html:
#                        candidate_html = bnode.html
#
#            # b) fallback: body container directly from full document
#            if candidate_html is None:
#                bnode = doc.css_first(bc_css)
#                if bnode and bnode.html:
#                    candidate_html = bnode.html
#
#            if candidate_html:
#                container_html = candidate_html
#                ordered_text = _container_text_with_exclusions_for_extract(
#                    "<div>" + container_html + "</div>", "div", excludes
#                )
#                if ordered_text:
#                    ordered_text = _apply_boilerplate(ordered_text, boiler)


        # ---------- Legacy fallback: body_html / fallback join ----------
        if ordered_text is None:
            # If body_html is empty but a body_fallback exists, try it
            if not res.get("body_html") and isinstance(rules.get("body_fallback"), dict) and "join" in rules["body_fallback"]:
                try:
                    bf_sel = rules["body_fallback"]["join"]
                    res["body_html"] = _sel_join(doc, bf_sel)
                except Exception:
                    pass

            raw = res.get("body_html") or ""
            if raw:
                # cheap plain strip when no new config present
                from conty_core.normalize import clean_text as _ct
                ordered_text = _ct(raw, strip_html=True)
                container_html = container_html or res.get("container_html")

        body_len = len(ordered_text or "")

        # ---------- Acceptance ----------
        # Per-template min chars (strict) or lenient builder mode
        min_chars = _min_chars_from_template(t)
        title_ok = bool(res.get("title"))
        main_img_abs = _abs_url(res.get("main_image"), url)

        if lenient:
            if title_ok or body_len > 200:
                article = Article(
                    url=url,
                    site=site_cfg.get("site",""),
                    title=clean_text(res.get("title")),
                    subtitle=clean_text(res.get("subtitle")),
                    lead=clean_text(res.get("lead"), strip_html=True),
                    author=clean_text(res.get("author")),
                    section=clean_text(res.get("section")),
                    tags=res.get("tags") or [],
                    published_time=res.get("published_time"),
                    updated_time=res.get("updated_time"),
                    body_html=res.get("body_html"),
                    text=ordered_text,
                    canonical_url=res.get("canonical_url"),
                    main_image=main_img_abs,
                    template_used=t_name,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    status="ok",
                    container_html=container_html,
                )
                article = postprocess_article_obj(article)
                return article
            else:
                last_error_reason = "short_body" if body_len <= 200 else "no_title"
                continue

        # Strict / production acceptance (reworked to allow "brief" articles)
        # Threshold to classify brief vs full:
        threshold = int(min_chars) if min_chars is not None else 400

        # Accept if the article has a title (always save), regardless of body length.
        # classify brief/full downstream (in Run_Scrapes) by comparing body_len to `threshold`.
        if title_ok:
            article = Article(
                url=url,
                site=site_cfg.get("site",""),
                title=clean_text(res.get("title")),
                subtitle=clean_text(res.get("subtitle")),
                lead=clean_text(res.get("lead"), strip_html=True),
                author=clean_text(res.get("author")),
                section=clean_text(res.get("section")),
                tags=res.get("tags") or [],
                published_time=res.get("published_time"),
                updated_time=res.get("updated_time"),
                body_html=res.get("body_html"),
                text=ordered_text,
                canonical_url=res.get("canonical_url"),
                main_image=main_img_abs,
                template_used=t_name,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                status="ok",
                container_html=container_html,
            )
            article = postprocess_article_obj(article)
            # Attach brief metrics
            try:
                setattr(article, "body_chars", body_len)
                setattr(article, "min_body_chars", threshold)
                setattr(article, "is_brief", bool(body_len < threshold))
            except Exception:
                pass
            return article
        else:
            # No title → this template is a miss - continue to the next template
            last_error_reason = "no_title"
            continue

    # nothing matched/accepted
    return Article(
        url=url,
        site=site_cfg.get("site",""),
        status="extract_error",
        error=last_error_reason or "No template matched",
        template_used=last_template_tried or "no_match",
    )

def extract_and_postprocess(
    url: str,
    html: str,
    site_cfg: dict,
    *,
    forced_template: str | None = None,
    lenient: bool = False,
) -> dict:
    """
    Canonical extraction pipeline for the app.

    - Calls extract_article(...), which already:
        * picks a template
        * extracts fields
        * builds an Article
        * runs postprocess_article_obj(...) on it
    - Returns a plain dict with all Article fields.
    - Ensures the existance of:
        * published_datetime / updated_datetime (aliases for *_time)
        * published_date / updated_date (YYYY-MM-DD slices).
    """
    art = extract_article(
        url=url,
        html=html,
        site_cfg=site_cfg,
        forced_template=forced_template,
        lenient=lenient,
    )

    # Normal case: Article dataclass
    if isinstance(art, Article):
        data = asdict(art)
    # Fallback: if for any reason a dict is returned
    elif isinstance(art, dict):
        data = dict(art)
    else:
        # Extremely defensive: unknown type
        data = {"status": getattr(art, "status", "extract_error")}

    # ---- Unify datetime keys ----
    # published_time/updated_time already normalized by postprocess_article_obj
    if "published_time" in data and "published_datetime" not in data:
        data["published_datetime"] = data.get("published_time")
    if "updated_time" in data and "updated_datetime" not in data:
        data["updated_datetime"] = data.get("updated_time")

    # Derive date-only fields from the normalized datetimes if missing
    pub_dt = data.get("published_datetime")
    if isinstance(pub_dt, str) and len(pub_dt) >= 10 and not data.get("published_date"):
        data["published_date"] = pub_dt[:10]

    upd_dt = data.get("updated_datetime")
    if isinstance(upd_dt, str) and len(upd_dt) >= 10 and not data.get("updated_date"):
        data["updated_date"] = upd_dt[:10]

    return data

