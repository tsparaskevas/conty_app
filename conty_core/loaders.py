from pathlib import Path
import pandas as pd
from typing import List, Optional

def list_csvs_in_folder(folder: str) -> List[Path]:
    p = Path(folder).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return []
    return sorted(p.glob("*.csv"))

def read_urls_from_csv(csv_path: Path, url_col: Optional[str] = None, max_rows: int = 20000) -> List[str]:
    df = pd.read_csv(csv_path)
    if url_col is None:
        candidates = [c for c in df.columns if c.lower() == "url"] or list(df.columns)
        url_col = candidates[0]
    urls = df[url_col].dropna().astype(str).head(max_rows).tolist()
    urls = [u.strip() for u in urls if u.strip().startswith(("http://","https://"))]
    return urls
