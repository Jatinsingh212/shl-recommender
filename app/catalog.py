from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT_DIR / "data" / "catalog.json"
BASE_URL = "https://www.shl.com/products/product-catalog/"
LISTING_PAGE_SIZE = 12
MAX_START = 372

TOKEN_RE = re.compile(r"[a-z0-9+#/.]{2,}")
TABLE_RE = re.compile(r"<table[^>]*>[\s\S]*?</table>", re.IGNORECASE)
ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
CELL_RE = re.compile(r"<td[^>]*>([\s\S]*?)</td>", re.IGNORECASE)
TYPE_RE = re.compile(r"product-catalogue__key[^>]*>([^<]+)<", re.IGNORECASE)
LINK_RE = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>', re.IGNORECASE)

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


@dataclass(frozen=True)
class Assessment:
    name: str
    url: str
    test_type: str
    test_type_label: str
    description: str
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    assessment_length: str
    remote_testing: bool
    adaptive_irt: bool
    fact_sheet_urls: tuple[str, ...]
    searchable_text: str
    tokens: frozenset[str]


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    return normalize_text(html.unescape(re.sub(r"<[^>]+>", " ", text)))


def _read_catalog_payload() -> dict:
    if not CATALOG_PATH.exists():
        return {"assessments": []}

    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if payload.get("assessments"):
        return payload
    return {"assessments": []}


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; SHLAssignmentBot/1.0)"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_listing_rows(html_text: str) -> list[dict]:
    tables = TABLE_RE.findall(html_text)
    table = next(
        (item for item in tables if re.search(r"<th[^>]*>\s*Individual Test Solutions\s*</th>", item, re.IGNORECASE)),
        "",
    )
    if not table:
        return []

    assessments: list[dict] = []
    for row in ROW_RE.findall(table)[1:]:
        cells = CELL_RE.findall(row)
        if len(cells) < 4:
            continue

        link_match = LINK_RE.search(cells[0])
        if not link_match:
            continue

        type_codes = "".join(_strip_html(match) for match in TYPE_RE.findall(cells[3]))
        assessment = {
            "name": _strip_html(link_match.group(2)),
            "url": urljoin(BASE_URL, link_match.group(1)),
            "test_type": type_codes,
            "test_type_label": ", ".join(TEST_TYPE_LABELS.get(code, code) for code in type_codes),
            "description": "",
            "job_levels": [],
            "languages": [],
            "assessment_length": "",
            "remote_testing": "catalogue__circle -yes" in cells[1],
            "adaptive_irt": "catalogue__circle -yes" in cells[2],
            "fact_sheet_urls": [],
        }
        assessment["searchable_text"] = normalize_text(
            " ".join([assessment["name"], assessment["test_type"], assessment["test_type_label"]])
        )
        assessments.append(assessment)

    return assessments


def _fetch_catalog_payload() -> dict:
    assessments: list[dict] = []
    for start in range(0, MAX_START + LISTING_PAGE_SIZE, LISTING_PAGE_SIZE):
        page = _fetch_text(f"{BASE_URL}?start={start}&type=1")
        page_items = _extract_listing_rows(page)
        if not page_items:
            break
        urls = {item["url"] for item in assessments}
        new_items = [item for item in page_items if item["url"] not in urls]
        if not new_items:
            break
        assessments.extend(new_items)

    payload = {
        "generated_at": normalize_text("generated from live SHL catalog"),
        "source": BASE_URL,
        "total_assessments": len(assessments),
        "assessments": sorted(assessments, key=lambda item: item["name"]),
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _payload_to_assessments(payload: dict) -> list[Assessment]:
    assessments: list[Assessment] = []
    for raw in payload.get("assessments", []):
        searchable_text = normalize_text(raw.get("searchable_text", ""))
        token_source = " ".join(
            [
                raw["name"],
                raw.get("description", ""),
                searchable_text,
                " ".join(raw.get("job_levels", [])),
                " ".join(raw.get("languages", [])),
            ]
        )
        assessments.append(
            Assessment(
                name=raw["name"],
                url=raw["url"],
                test_type=raw["test_type"],
                test_type_label=raw.get("test_type_label", raw["test_type"]),
                description=raw.get("description", ""),
                job_levels=tuple(raw.get("job_levels", [])),
                languages=tuple(raw.get("languages", [])),
                assessment_length=raw.get("assessment_length", ""),
                remote_testing=bool(raw.get("remote_testing", False)),
                adaptive_irt=bool(raw.get("adaptive_irt", False)),
                fact_sheet_urls=tuple(raw.get("fact_sheet_urls", [])),
                searchable_text=searchable_text,
                tokens=frozenset(tokenize(token_source)),
            )
        )
    return assessments


@lru_cache(maxsize=1)
def load_catalog() -> list[Assessment]:
    payload = _read_catalog_payload()
    if not payload.get("assessments"):
        payload = _fetch_catalog_payload()
    return _payload_to_assessments(payload)


def find_by_name_fragment(query: str, pool: Iterable[Assessment] | None = None) -> list[Assessment]:
    assessments = list(pool if pool is not None else load_catalog())
    normalized_query = normalize_text(query).lower()
    if not normalized_query:
        return []

    exact = [
        item
        for item in assessments
        if normalized_query == item.name.lower() or normalized_query in item.name.lower()
    ]
    if exact:
        return exact

    query_tokens = tokenize(normalized_query)
    if not query_tokens:
        return []

    matches: list[tuple[int, Assessment]] = []
    for item in assessments:
        overlap = len(query_tokens & item.tokens)
        if overlap:
            matches.append((overlap, item))

    matches.sort(key=lambda pair: (-pair[0], pair[1].name))
    return [item for _, item in matches[:10]]
