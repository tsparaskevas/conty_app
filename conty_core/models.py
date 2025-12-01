from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["Article"]

@dataclass
class Article:
    url: str
    site: str
    title: Optional[str] = None
    subtitle: Optional[str] = None
    lead: Optional[str] = None
    author: Optional[str] = None
    published_time: Optional[str] = None
    updated_time: Optional[str] = None
    container_html: Optional[str] = None
    body_html: Optional[str] = None
    text: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    section: Optional[str] = None
    canonical_url: Optional[str] = None
    main_image: Optional[str] = None
    images: List[str] = field(default_factory=list)
    breadcrumbs: List[str] = field(default_factory=list)
    paywalled: Optional[bool] = None
    template_used: Optional[str] = None
    fetched_at: Optional[str] = None
    status: str = "ok"
    error: Optional[str] = None
