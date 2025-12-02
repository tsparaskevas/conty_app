---
title: 'Conty App: a configurable framework for reproducible news-article scraping'
tags:
  - Python
  - web scraping
  - journalism studies
  - computational social science
  - text mining
authors:
  - name: Thodoris Paraskevas
    orcid: 0000-0001-8857-7581 
    affiliation: 1
affiliations:
  - name: Faculty of Communication and Media Studies, National and Kapodistrian University of Athens, Greece
    index: 1
date: 2025-12-01
bibliography: paper.bib
---

# Summary

Online news articles are central data sources in journalism studies, political
communication, and computational social science. Yet collecting such data
requires building reliable scrapers for multiple news outlets, each with
distinct HTML structures, dynamic elements, and cookie/consent interactions.
Researchers often rely on ad-hoc scripts that are fragile, non-transparent,
and difficult to adapt or share.

**Conty App** is an interactive, Streamlit-based environment for configuring,
testing, and running article-level scrapers for news websites. It builds on
a reusable Python extraction engine (**`conty_core`**), providing a graphical
interface for specifying template rules, previewing extraction results,
running multi-site scrapes, and inspecting the resulting article datasets.
The system aims to make data collection **reproducible, inspectable,
configurable, and shareable**.

# Statement of need

Researchers frequently require structured corpora of news articles for
examining media coverage, event-based reporting, longitudinal trends,
framing, agenda-setting, or computational text analysis. However, modern news
sites feature heterogeneous page layouts, dynamic content, embedded media,
and consent/cookie pop-ups, making scraping technically demanding.
Existing solutions—generic web-scraping libraries, one-off scripts, or
site-specific scrapers—are typically insufficient because they:

1. lack **site-configurable extraction templates**, 
2. do not provide **interactive previewing and validation**, 
3. rarely include **consent-handling or Requests→Selenium fallback**, 
4. intertwine data collection with analysis, impeding **reproducibility**, and 
5. offer little support for **multi-site workflows**.

**Conty App** fills this gap by combining a configurable template system with
an interactive interface for inspecting extraction logic, testing it against
real HTML, and running reproducible article scrapes via a unified workflow.
By separating the interface (`conty_app`) from the core engine (`conty_core`),
the software supports both graphical (UI-based) and programmatic usage, making
it accessible to researchers with diverse technical backgrounds.

# Software description

## Functionality

Conty App supports the full workflow from URL lists to structured article
tables:

### 1. Template configuration (build/edit/test scrapers)
Users define YAML scraper templates specifying:

- CSS selectors for fields such as **title**, **body**, **author**, **date**,
  **section**, **tags**, **summary**, or **paywall**;
- site-level settings (base URL, teaser fields, date parsing hints);
- boilerplate removal rules;
- optional consent/cookie XPaths; and
- nested container logic for complex body extraction.

The UI offers:

- a “Fetch page” tool to load HTML (or snapshots in demo mode),
- interactive previews of extracted fields,
- live highlighting of missing/failed selectors, and
- a **matrix tester** that evaluates templates against teaser-CSV URLs.

### 2. Article scraping (run scrapes)
Given one or more teaser CSVs containing article URLs, Conty App:

- selects the appropriate template for each URL,
- fetches pages via **Requests → Selenium fallback** with optional
  consent-click handling,
- extracts article fields with `conty_core.extract`, and
- writes article tables and per-run logs to disk.

Both **single-site** and **multi-site** runs are supported. Extraction errors
are recorded in structured logs to aid debugging and reproducibility.

### 3. Data inspection (View / Explore data)
The application includes tools for inspecting and exploring results:

- browse article tables with filtering,
- inspect raw run logs,
- view basic descriptive summaries such as counts by site or over time.

### 4. Demo mode (for safe public deployment)
A dedicated demo mode (`CONTY_DEMO=1`) ships with example HTML snapshots,
teaser CSVs, and precomputed article tables stored under `data/demo/`. In
this mode:

- **no network requests** are performed,
- **no files are written**, and
- extraction previews and run results use demo data.

This enables safe hosting on Streamlit Community Cloud while demonstrating the
entire workflow.

# Relation to Teasy App

Conty App complements a companion tool, **Teasy App**, which focuses on
scraping *teaser pages* (front pages, category pages, opinion pages, or search
results) to collect article URLs and metadata. While Teasy App operates at the
teaser level, Conty App processes the corresponding **full articles** using
site-specific templates. Together, the two applications form a complete,
modular, and reproducible pipeline for collecting large-scale news datasets,
but each tool is designed to function independently and addresses a distinct
stage of the data-collection workflow.

# Acknowledgements

I thank colleagues and collaborators who provided feedback on early scraper
designs and test corpora, and the maintainers of the open-source libraries on
which this software depends, including Streamlit [@streamlit2021], Requests
[@requests], BeautifulSoup [@bs4], Selenium [@selenium], Pandas [@pandas],
and lxml [@lxml].

# References

