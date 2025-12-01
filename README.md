# Conty App

UI to build/edit/test **article scrapers** for news-sites, run single/multi-site article scrapes, inspect extraction results, and browse collected data.

- Build scrapers with manual CSS selectors (title, body, author, date, section, tags, summary, paywall)
- Requests → Selenium fallback with consent-click support
- Template system: site-level + multiple template variants per site
- Article extraction via container + nested container-text model
- Matrix tester for validating selectors against teaser CSVs
- Multi-site runs, per-run logs, saved HTML snapshots
- Scraped article browser & basic exploration tools
- **Demo mode** for read-only public use (no network, no writing to disk)

## Quick start (local)

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run conty_app/Home.py
```

## Citation

If you use this app, please cite:

Paraskevas, T. (2025). *Conty App* (Version 1.0.0) [Software]. GitHub. https://github.com/tsparaskevas/conty_app

```bibtex
@software{Paraskevas_Conty_App_2025,
  author  = {Paraskevas, Thodoris},
  title   = {Conty App},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/tsparaskevas/conty_app},
  note    = {Software}
}
```

## Live demo (Streamlit Community Cloud)

👉 Try the read-only demo: **[Conty App — Streamlit Cloud](https://contyapp-znynnrnbd8fmbrmtqgt5nm.streamlit.app/)**

**What works in the demo**
- View templates and example extraction previews
- Browse sample article datasets
- Inspect demo run logs
- Matrix-test example URLs

**What’s disabled (and why)**
- No Requests/Selenium fetching
- No writing updates to data/scrapers/
- No file output under data/articles/ or data/run_logs/
- Reason: public cloud sessions are resource-limited, storage is ephemeral, and many news sites actively block or rate-limit headless scrapers. 
- The demo sets `TEASY_DEMO=1`, which disables network fetches. Clone the repo to run full scrapes locally.

**Run the full app locally**
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run conty_app/Home.py
# Scraping is enabled locally
```

## License
MIT - see the [LICENSE](LICENSE) file for details.

© 2025 Thodoris Paraskevas
