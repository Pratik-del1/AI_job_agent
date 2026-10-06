"""Job identity, duplicates, the record of seen jobs, and rank order."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit


def normalize_url(url: str) -> str:
    """URL without query, fragment, trailing slash or a final /apply."""

    parts = urlsplit(str(url or "").strip())
    if not parts.netloc:
        return ""

    path = parts.path.rstrip("/")
    if path.lower().endswith("/apply"):
        path = path[: -len("/apply")]

    return f"{parts.netloc.lower()}{path}".lower()


def job_key(job: dict) -> str:
    """Stable identity of a posting: its ID, else its URL, else a hash of
    title, company and location."""

    job_id = str(job.get("job_id") or "").strip()
    if job_id and job_id.lower() != "nan":
        return f"id:{job_id}"

    url = normalize_url(job.get("apply_url", ""))
    if url:
        return f"url:{url}"

    basis = "|".join(
        str(job.get(field) or "").strip().lower()
        for field in ("title", "company", "location")
    )
    return "hash:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


LOCATION_SEPARATOR = " / "


def dedupe_jobs(jobs: list[dict]) -> tuple[list[dict], int]:
    """One record per posting. Returns (unique jobs, records merged away).

    Two records are the same posting when they share an ID or a URL. Some
    feeds list one posting once per city: those are merged into a single
    record whose location names every city, so no location is lost. The
    same role posted with its own ID and URL is a separate posting and is
    kept.
    """

    by_key: dict[str, dict] = {}
    key_by_url: dict[str, str] = {}

    for job in jobs:
        key = job_key(job)
        url = normalize_url(job.get("apply_url", ""))

        existing_key = key if key in by_key else key_by_url.get(url)

        if existing_key is None:
            by_key[key] = dict(job)
            if url:
                key_by_url[url] = key
            continue

        kept = by_key[existing_key]
        locations = [
            part
            for part in str(kept.get("location") or "").split(
                LOCATION_SEPARATOR
            )
            if part
        ]
        location = str(job.get("location") or "").strip()
        if location and location.lower() not in {
            part.lower() for part in locations
        }:
            kept["location"] = LOCATION_SEPARATOR.join(
                [*locations, location]
            )

    unique = list(by_key.values())
    return unique, len(jobs) - len(unique)


class SeenJobs:
    """Every job key ever fetched, with when it was first seen.

    "New" means absent from this record, not absent from the last
    recommendation list. On the very first run there is no history, so
    nothing is called new.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.has_history = self.path.exists()
        self.first_seen: dict[str, str] = {}

        if self.has_history:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.first_seen = dict(data.get("jobs", {}))

    def is_new(self, key: str) -> bool:
        return self.has_history and key not in self.first_seen

    def record(self, keys, now: Optional[datetime] = None) -> None:
        stamp = (now or datetime.now()).isoformat(timespec="seconds")
        for key in keys:
            self.first_seen.setdefault(key, stamp)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": 1, "jobs": self.first_seen},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.has_history = True


def sort_key(overall_score: float, is_new: bool, key: str, tolerance: float):
    """Best match first. Scores within ``tolerance`` share a band, and only
    inside a band does a new job go ahead of an old one."""

    band = math.floor(overall_score / tolerance + 0.5)
    return (-band, 0 if is_new else 1, -overall_score, key)
