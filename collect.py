"""Collect raw demand and supply data for candidate content niches.

Sources (all free):

* Wikipedia Pageviews REST API -- monthly human pageviews per article.
* Google Trends via pytrends -- 12-month and 5-year interest, rising queries.
* YouTube Data API v3 -- videos per keyword in the last N days vs the same
  window a year earlier, plus view and subscriber statistics.
* Reddit JSON (public, or app-only OAuth if credentials are set) -- subscribers
  and posting rate per subreddit.

Every source writes its own CSV under ``data/raw``. A failing source never
stops the others: failures are written to ``data/raw/errors.csv`` and nothing is
filled in or invented in their place.

Usage::

    python collect.py --sources wiki,trends,youtube,reddit
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import logging
import os
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote

import requests
import yaml

LOG = logging.getLogger("niche-scout")

USER_AGENT = (
    "niche-scout/0.1 (+https://github.com/aseembhalla/niche-scout) " "python-requests"
)
REDDIT_USER_AGENT = (
    "python:niche-scout:0.1 (+https://github.com/aseembhalla/niche-scout)"
)
WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}"
)
YT_API = "https://www.googleapis.com/youtube/v3"
ALL_SOURCES = ("wiki", "trends", "trends_solo", "youtube", "reddit")

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Configuration and shared helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Niche:
    """One candidate niche and the inputs used to measure it."""

    id: str
    name: str
    wiki: list[str]
    keywords: list[str]
    subreddits: list[str]


@dataclass
class Config:
    """Parsed ``niches.yaml``."""

    niches: list[Niche]
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Config:
        """Read and validate the YAML config file."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        niches = [Niche(**item) for item in raw["niches"]]
        ids = [n.id for n in niches]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate niche ids in config")
        return cls(niches=niches, settings=raw.get("settings", {}))


class Store:
    """Writes CSV outputs and the shared error log."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
        self._errors: list[dict[str, str]] = []

    def write(self, name: str, rows: Sequence[dict[str, Any]]) -> Path:
        """Overwrite ``name`` with ``rows`` (git history keeps old runs)."""
        path = self.out_dir / name
        if not rows:
            LOG.warning("No rows for %s; file not written", name)
            return path
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOG.info("Wrote %d rows to %s", len(rows), path)
        return path

    def error(self, source: str, niche: str, item: str, message: str) -> None:
        """Record a failed fetch so the gap is visible, never silently zero."""
        LOG.error("[%s] %s / %s: %s", source, niche, item, message)
        self._errors.append(
            {
                "source": source,
                "niche_id": niche,
                "item": item,
                "error": message[:500],
                "fetched_at": self.run_at,
            }
        )

    def flush_errors(self, sources: Iterable[str]) -> None:
        """Merge this run's errors into errors.csv, replacing re-run sources."""
        path = self.out_dir / "errors.csv"
        rerun = set(sources)
        kept: list[dict[str, str]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                kept = [r for r in csv.DictReader(handle) if r["source"] not in rerun]
        rows = kept + self._errors
        fields = ["source", "niche_id", "item", "error", "fetched_at"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def make_session(user_agent: str = USER_AGENT) -> requests.Session:
    """Return a requests session with a descriptive User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def with_retries(
    func: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 5.0,
    label: str = "",
) -> T:
    """Call ``func`` with exponential backoff and jitter.

    Raises the last exception if every attempt fails.
    """
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - re-raised after last attempt
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, base_delay)
            LOG.warning(
                "%s failed (attempt %d/%d): %s; sleeping %.0fs",
                label,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    """GET ``url`` and return parsed JSON, raising for HTTP errors."""
    resp = session.get(url, params=params, timeout=timeout)
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"HTTP {resp.status_code} for {resp.url}: {resp.text[:200]}",
            response=resp,
        )
    return resp.json()


def last_complete_month(today: dt.date) -> dt.date:
    """Return the first day of the last fully finished month."""
    first_this_month = today.replace(day=1)
    return (first_this_month - dt.timedelta(days=1)).replace(day=1)


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #


def resolve_wiki_titles(
    session: requests.Session, titles: Sequence[str]
) -> dict[str, str | None]:
    """Map each input title to its canonical article title.

    Follows normalisation and redirects. Missing articles map to ``None``.
    The Pageviews API counts a redirect separately from its target, so the
    canonical title must be used to get the article's real traffic.
    """
    resolved: dict[str, str | None] = {}
    for start in range(0, len(titles), 50):
        chunk = list(titles[start : start + 50])
        data = with_retries(
            partial(
                get_json,
                session,
                WIKI_ACTION_API,
                {
                    "action": "query",
                    "titles": "|".join(chunk),
                    "redirects": 1,
                    "format": "json",
                    "formatversion": 2,
                },
            ),
            label="wiki resolve",
        )
        query = data.get("query", {})
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        missing = {
            p["title"] for p in query.get("pages", []) if p.get("missing", False)
        }
        for title in chunk:
            name = normalized.get(title, title)
            name = redirects.get(name, name)
            resolved[title] = None if name in missing else name
    return resolved


def collect_wiki(config: Config, store: Store) -> None:
    """Fetch monthly human pageviews for every configured article."""
    session = make_session()
    start = dt.date.fromisoformat(config.settings.get("wiki_start", "2016-01-01"))
    end_month = last_complete_month(dt.date.today())
    end_day = (end_month.replace(day=28) + dt.timedelta(days=4)).replace(
        day=1
    ) - dt.timedelta(days=1)

    all_titles = sorted({t for n in config.niches for t in n.wiki})
    resolved = resolve_wiki_titles(session, all_titles)

    rows: list[dict[str, Any]] = []
    for niche in config.niches:
        seen: set[str] = set()
        for title in niche.wiki:
            target = resolved.get(title)
            if target is None:
                store.error("wiki", niche.id, title, "article not found")
                continue
            if target in seen:
                store.error("wiki", niche.id, title, f"duplicate of {target}; skipped")
                continue
            seen.add(target)
            url = WIKI_PAGEVIEWS_API.format(
                title=quote(target.replace(" ", "_"), safe=""),
                start=start.strftime("%Y%m%d00"),
                end=end_day.strftime("%Y%m%d00"),
            )
            try:
                data = with_retries(
                    partial(get_json, session, url), label=f"wiki {target}"
                )
            except Exception as exc:  # noqa: BLE001 - logged, run continues
                store.error("wiki", niche.id, title, str(exc))
                continue
            for item in data.get("items", []):
                rows.append(
                    {
                        "niche_id": niche.id,
                        "article_input": title,
                        "article": target,
                        "month": f"{item['timestamp'][:4]}-{item['timestamp'][4:6]}",
                        "views": item["views"],
                        "source_url": url,
                        "fetched_at": store.run_at,
                    }
                )
            time.sleep(0.2)
    store.write("wiki_monthly.csv", rows)


# --------------------------------------------------------------------------- #
# Google Trends
# --------------------------------------------------------------------------- #


def _trends_call(func: Callable[[], T], label: str) -> T:
    """Run a pytrends call with long backoff, then pause politely."""
    result = with_retries(func, attempts=5, base_delay=60.0, label=label)
    time.sleep(random.uniform(15, 30))
    return result


def _trends_client(store: Store) -> Any:
    """Create a pytrends client, or log an error and return ``None``.

    The constructor fetches a Google cookie, so it can fail or be rate-limited.
    Do not pass retries/backoff_factor: pytrends 4.9.2 breaks with urllib3>=2.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:  # pragma: no cover - dependency missing
        store.error("trends", "*", "pytrends", f"import failed: {exc}")
        return None
    try:
        return with_retries(
            partial(TrendReq, hl="en-US", tz=-330, timeout=(10, 30)),
            attempts=4,
            base_delay=60.0,
            label="trends session",
        )
    except Exception as exc:  # noqa: BLE001
        store.error("trends", "*", "session", str(exc))
        return None


def collect_trends(config: Config, store: Store) -> None:
    """Fetch Google Trends interest and rising queries for all keywords.

    Keywords are queried in batches of four plus one shared anchor keyword,
    so values from different batches can be put on a common scale
    (``value / mean(anchor)``). Growth ratios use each keyword's own series,
    which the batch scaling does not affect.
    """
    geo = str(config.settings.get("trends_geo", ""))
    anchor = str(config.settings.get("trends_anchor", "sudoku"))
    kw_to_niche: dict[str, str] = {}
    for niche in config.niches:
        for kw in niche.keywords:
            kw_to_niche.setdefault(kw, niche.id)
    keywords = list(kw_to_niche)
    batches = [keywords[i : i + 4] for i in range(0, len(keywords), 4)]

    client = _trends_client(store)
    if client is None:
        return
    series_rows: list[dict[str, Any]] = []
    rising_rows: list[dict[str, Any]] = []
    source = "Google Trends via pytrends 4.9.2 (unofficial)"

    for batch_no, batch in enumerate(batches, start=1):
        payload = [*batch, anchor]
        for timeframe in ("today 12-m", "today 5-y"):
            label = f"trends batch {batch_no} {timeframe}"
            try:

                def fetch(tf: str = timeframe) -> Any:
                    client.build_payload(payload, timeframe=tf, geo=geo)
                    return client.interest_over_time()

                frame = _trends_call(fetch, label)
            except Exception as exc:  # noqa: BLE001 - logged, run continues
                for kw in batch:
                    store.error("trends", kw_to_niche[kw], kw, f"{timeframe}: {exc}")
                continue
            if frame is None or frame.empty:
                for kw in batch:
                    store.error("trends", kw_to_niche[kw], kw, f"{timeframe}: empty")
                continue
            for date, rec in frame.iterrows():
                for kw in batch:
                    series_rows.append(
                        {
                            "niche_id": kw_to_niche[kw],
                            "keyword": kw,
                            "timeframe": timeframe,
                            "date": date.date().isoformat(),
                            "value": int(rec[kw]),
                            "anchor": anchor,
                            "anchor_value": int(rec[anchor]),
                            "is_partial": bool(rec.get("isPartial", False)),
                            "batch": batch_no,
                            "geo": geo or "WORLD",
                            "source": source,
                            "fetched_at": store.run_at,
                        }
                    )

            if timeframe != "today 12-m":
                continue
            try:
                related = _trends_call(client.related_queries, f"{label} related")
            except Exception as exc:  # noqa: BLE001
                for kw in batch:
                    store.error("trends", kw_to_niche[kw], kw, f"related: {exc}")
                continue
            for kw in batch:
                rising = (related or {}).get(kw, {}).get("rising")
                if rising is None or rising.empty:
                    continue
                for _, rec in rising.iterrows():
                    rising_rows.append(
                        {
                            "niche_id": kw_to_niche[kw],
                            "keyword": kw,
                            "rising_query": rec["query"],
                            "rise": rec["value"],
                            "timeframe": "today 12-m",
                            "geo": geo or "WORLD",
                            "source": source,
                            "fetched_at": store.run_at,
                        }
                    )

    store.write("trends_timeseries.csv", series_rows)
    store.write("trends_rising.csv", rising_rows)


def collect_trends_solo(config: Config, store: Store) -> None:
    """Fetch each keyword's 5-year Trends series on its own.

    Alone in a request, a keyword is scaled to its own peak (100), so small
    keywords keep full resolution. Use these series for growth ratios; use
    the anchored batches in ``trends_timeseries.csv`` only for size.
    """
    client = _trends_client(store)
    if client is None:
        return
    geo = str(config.settings.get("trends_geo", ""))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for niche in config.niches:
        for kw in niche.keywords:
            if kw in seen:
                continue
            seen.add(kw)
            try:

                def fetch(word: str = kw) -> Any:
                    client.build_payload([word], timeframe="today 5-y", geo=geo)
                    return client.interest_over_time()

                frame = _trends_call(fetch, f"trends solo {kw}")
            except Exception as exc:  # noqa: BLE001
                store.error("trends_solo", niche.id, kw, str(exc))
                continue
            if frame is None or frame.empty:
                store.error("trends_solo", niche.id, kw, "empty (too little volume)")
                continue
            for date, rec in frame.iterrows():
                rows.append(
                    {
                        "niche_id": niche.id,
                        "keyword": kw,
                        "date": date.date().isoformat(),
                        "value": int(rec[kw]),
                        "is_partial": bool(rec.get("isPartial", False)),
                        "geo": geo or "WORLD",
                        "source": "Google Trends via pytrends 4.9.2, keyword alone",
                        "fetched_at": store.run_at,
                    }
                )
    store.write("trends_solo.csv", rows)


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #


class QuotaExceeded(Exception):
    """Raised when the YouTube daily quota is exhausted."""


def _yt_get(session: requests.Session, path: str, params: dict[str, Any]) -> Any:
    """Call a YouTube Data API endpoint, translating quota errors."""
    resp = session.get(f"{YT_API}/{path}", params=params, timeout=30)
    if resp.status_code == 403 and "quota" in resp.text.lower():
        raise QuotaExceeded(resp.text[:200])
    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _chunks(items: Sequence[str], size: int = 50) -> Iterable[list[str]]:
    """Yield successive ``size``-long lists from ``items``."""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _median(values: Sequence[float]) -> float | None:
    """Median, or ``None`` for an empty sequence."""
    return statistics.median(values) if values else None


def collect_youtube(config: Config, store: Store) -> None:
    """Count and sample videos per keyword for two matched time windows.

    Windows: the last ``youtube_window_days`` days, and the same span one
    year earlier (season-matched). Quota per keyword/window: 100 units for
    search.list plus 1 each for videos.list and channels.list.
    """
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        store.error("youtube", "*", "YOUTUBE_API_KEY", "environment variable not set")
        return

    session = make_session()
    days = int(config.settings.get("youtube_window_days", 90))
    lang = str(config.settings.get("youtube_relevance_language", "en"))
    min_subs = int(config.settings.get("min_channel_subs_for_ratio", 1000))
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    windows = {
        "recent": (now - dt.timedelta(days=days), now),
        "prior_year": (
            now - dt.timedelta(days=365 + days),
            now - dt.timedelta(days=365),
        ),
    }

    window_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    try:
        for niche in config.niches:
            for kw in niche.keywords:
                for window, (after, before) in windows.items():
                    try:
                        search = _yt_get(
                            session,
                            "search",
                            {
                                "part": "snippet",
                                "q": kw,
                                "type": "video",
                                "maxResults": 50,
                                "publishedAfter": after.isoformat().replace(
                                    "+00:00", "Z"
                                ),
                                "publishedBefore": before.isoformat().replace(
                                    "+00:00", "Z"
                                ),
                                "relevanceLanguage": lang,
                                "key": key,
                            },
                        )
                        ids = [
                            i["id"]["videoId"]
                            for i in search.get("items", [])
                            if i.get("id", {}).get("videoId")
                        ]
                        stats: dict[str, dict[str, Any]] = {}
                        for chunk in _chunks(ids):
                            data = _yt_get(
                                session,
                                "videos",
                                {
                                    "part": "statistics,snippet",
                                    "id": ",".join(chunk),
                                    "key": key,
                                },
                            )
                            for item in data.get("items", []):
                                stats[item["id"]] = item
                        channel_ids = sorted(
                            {v["snippet"]["channelId"] for v in stats.values()}
                        )
                        subs: dict[str, int | None] = {}
                        for chunk in _chunks(channel_ids):
                            data = _yt_get(
                                session,
                                "channels",
                                {
                                    "part": "statistics",
                                    "id": ",".join(chunk),
                                    "key": key,
                                },
                            )
                            for item in data.get("items", []):
                                st = item.get("statistics", {})
                                hidden = st.get("hiddenSubscriberCount", False)
                                subs[item["id"]] = (
                                    None
                                    if hidden
                                    else int(st.get("subscriberCount", 0))
                                )
                    except QuotaExceeded:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        store.error("youtube", niche.id, f"{kw} [{window}]", str(exc))
                        continue

                    views: list[float] = []
                    per_day: list[float] = []
                    per_sub: list[float] = []
                    for vid in ids:
                        item = stats.get(vid)
                        if item is None:
                            continue
                        st = item.get("statistics", {})
                        published = dt.datetime.fromisoformat(
                            item["snippet"]["publishedAt"].replace("Z", "+00:00")
                        )
                        age_days = max((now - published).total_seconds() / 86400, 1.0)
                        n_views = int(st.get("viewCount", 0))
                        channel_subs = subs.get(item["snippet"]["channelId"])
                        views.append(n_views)
                        per_day.append(n_views / age_days)
                        if channel_subs is not None and channel_subs >= min_subs:
                            per_sub.append(n_views / channel_subs)
                        video_rows.append(
                            {
                                "niche_id": niche.id,
                                "keyword": kw,
                                "window": window,
                                "video_id": vid,
                                "channel_id": item["snippet"]["channelId"],
                                "channel_title": item["snippet"].get("channelTitle"),
                                "published_at": item["snippet"]["publishedAt"],
                                "views": n_views,
                                "likes": st.get("likeCount"),
                                "comments": st.get("commentCount"),
                                "channel_subs": channel_subs,
                                "age_days": round(age_days, 1),
                                "fetched_at": store.run_at,
                            }
                        )
                    window_rows.append(
                        {
                            "niche_id": niche.id,
                            "keyword": kw,
                            "window": window,
                            "published_after": after.date().isoformat(),
                            "published_before": before.date().isoformat(),
                            "total_results_est": search.get("pageInfo", {}).get(
                                "totalResults"
                            ),
                            "n_sampled": len(views),
                            "median_views": _median(views),
                            "median_views_per_day": _median(per_day),
                            "median_views_per_sub": _median(per_sub),
                            "n_channels_ratio": len(per_sub),
                            "source": "YouTube Data API v3 search.list "
                            "(totalResults is Google's estimate)",
                            "fetched_at": store.run_at,
                        }
                    )
    except QuotaExceeded as exc:
        store.error("youtube", "*", "quota", f"daily quota exhausted: {exc}")

    store.write("youtube_windows.csv", window_rows)
    store.write("youtube_videos.csv", video_rows)


# --------------------------------------------------------------------------- #
# Reddit
# --------------------------------------------------------------------------- #


class RedditClient:
    """Reads subreddit data via public JSON, falling back to app-only OAuth."""

    def __init__(self) -> None:
        self.session = make_session(REDDIT_USER_AGENT)
        self.client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
        self.oauth = False
        self.delay = 6.5  # unauthenticated access is limited to ~10 requests/min

    def _enable_oauth(self) -> bool:
        """Switch to OAuth if credentials exist. Returns True on success."""
        if self.oauth or not (self.client_id and self.client_secret):
            return False
        token = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        resp = self.session.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        self.session.headers["Authorization"] = f"Bearer {resp.json()['access_token']}"
        self.oauth = True
        self.delay = 1.0
        LOG.info("Reddit: switched to app-only OAuth")
        return True

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``path`` such as ``/r/python/about``; returns parsed JSON."""
        for _ in range(2):
            base = (
                "https://oauth.reddit.com" if self.oauth else "https://www.reddit.com"
            )
            url = f"{base}{path}" if self.oauth else f"{base}{path}.json"
            resp = self.session.get(url, params=params, timeout=30)
            time.sleep(self.delay)
            if resp.status_code in (401, 403, 429) and self._enable_oauth():
                continue
            if resp.status_code >= 400:
                raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
            return resp.json()
        raise requests.HTTPError(f"Reddit request failed for {path}")


def collect_reddit(config: Config, store: Store) -> None:
    """Fetch subscriber counts and recent posting rate per subreddit."""
    client = RedditClient()
    rows: list[dict[str, Any]] = []
    for niche in config.niches:
        for sub in niche.subreddits:
            try:
                about = client.get(f"/r/{sub}/about")["data"]
                listing = client.get(f"/r/{sub}/new", {"limit": 100})["data"]
            except Exception as exc:  # noqa: BLE001
                store.error("reddit", niche.id, sub, str(exc))
                continue
            created = sorted(
                float(c["data"]["created_utc"]) for c in listing.get("children", [])
            )
            span_days = (created[-1] - created[0]) / 86400 if len(created) > 1 else 0
            rows.append(
                {
                    "niche_id": niche.id,
                    "subreddit": sub,
                    "subscribers": about.get("subscribers"),
                    "active_users": about.get("active_user_count"),
                    "sub_created": dt.datetime.fromtimestamp(
                        float(about.get("created_utc", 0)), dt.UTC
                    ).date(),
                    "posts_sampled": len(created),
                    "sample_span_days": round(span_days, 2),
                    "posts_per_day": (
                        round((len(created) - 1) / span_days, 2) if span_days else None
                    ),
                    "access": "oauth" if client.oauth else "public-json",
                    "fetched_at": store.run_at,
                }
            )
    store.write("reddit.csv", rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

COLLECTORS: dict[str, Callable[[Config, Store], None]] = {
    "wiki": collect_wiki,
    "trends": collect_trends,
    "trends_solo": collect_trends_solo,
    "youtube": collect_youtube,
    "reddit": collect_reddit,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("niches.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--sources",
        default=",".join(ALL_SOURCES),
        help=f"Comma-separated subset of: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument(
        "--niches", default="", help="Optional comma-separated niche ids to run"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected collectors. Exit code 0 even if some fetches failed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    config = Config.load(args.config)
    if args.niches:
        wanted = {n.strip() for n in args.niches.split(",") if n.strip()}
        config.niches = [n for n in config.niches if n.id in wanted]

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        LOG.error("Unknown sources: %s", ", ".join(sorted(unknown)))
        return 2

    store = Store(args.out)
    for name in sources:
        LOG.info("=== %s ===", name)
        try:
            COLLECTORS[name](config, store)
        except Exception as exc:  # noqa: BLE001 - one source must not stop others
            store.error(name, "*", "collector", f"unhandled: {exc!r}")
    store.flush_errors(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
