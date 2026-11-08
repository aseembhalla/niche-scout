"""Offline tests: collectors run against fake HTTP responses, scorer on fixtures.

These only check the plumbing. No result from these tests is real data.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import collect
import score


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: Any, status: int = 200, url: str = "") -> None:
        self._payload = payload
        self.status_code = status
        self.text = str(payload)
        self.url = url

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _config() -> collect.Config:
    return collect.Config(
        niches=[
            collect.Niche(
                id="a",
                name="A",
                wiki=["Alpha", "Alpha redirect", "Missing page"],
                keywords=["kw a"],
                subreddits=["suba"],
            )
        ],
        settings={"wiki_start": "2016-01-01"},
    )


def test_wiki_resolves_redirects_and_logs_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_get(self: Any, url: str, params: Any = None, timeout: Any = None):
        if url == collect.WIKI_ACTION_API:
            return FakeResponse(
                {
                    "query": {
                        "redirects": [{"from": "Alpha redirect", "to": "Alpha"}],
                        "pages": [
                            {"title": "Alpha"},
                            {"title": "Missing page", "missing": True},
                        ],
                    }
                }
            )
        return FakeResponse(
            {"items": [{"timestamp": "2016010100", "views": 10}]}, url=url
        )

    monkeypatch.setattr("requests.Session.get", fake_get)
    monkeypatch.setattr(collect.time, "sleep", lambda _s: None)
    store = collect.Store(tmp_path)
    collect.collect_wiki(_config(), store)
    store.flush_errors(["wiki"])

    rows = list(csv.DictReader((tmp_path / "wiki_monthly.csv").open()))
    assert [r["article"] for r in rows] == ["Alpha"]  # redirect deduplicated
    errors = list(csv.DictReader((tmp_path / "errors.csv").open()))
    assert {e["item"] for e in errors} == {"Alpha redirect", "Missing page"}


def test_youtube_quota_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")

    def fake_get(self: Any, url: str, params: Any = None, timeout: Any = None):
        return FakeResponse({"error": "quotaExceeded"}, status=403)

    monkeypatch.setattr("requests.Session.get", fake_get)
    store = collect.Store(tmp_path)
    collect.collect_youtube(_config(), store)
    store.flush_errors(["youtube"])
    errors = list(csv.DictReader((tmp_path / "errors.csv").open()))
    assert errors[0]["item"] == "quota"


def test_youtube_window_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    published = (dt.datetime.now(dt.UTC) - dt.timedelta(days=10)).isoformat()

    def fake_get(self: Any, url: str, params: Any = None, timeout: Any = None):
        if url.endswith("/search"):
            return FakeResponse(
                {
                    "pageInfo": {"totalResults": 500},
                    "items": [{"id": {"videoId": "v1"}}, {"id": {"videoId": "v2"}}],
                }
            )
        if url.endswith("/videos"):
            return FakeResponse(
                {
                    "items": [
                        {
                            "id": vid,
                            "snippet": {
                                "channelId": "c1",
                                "publishedAt": published,
                            },
                            "statistics": {"viewCount": views},
                        }
                        for vid, views in (("v1", "1000"), ("v2", "3000"))
                    ]
                }
            )
        return FakeResponse(
            {"items": [{"id": "c1", "statistics": {"subscriberCount": "2000"}}]}
        )

    monkeypatch.setattr("requests.Session.get", fake_get)
    store = collect.Store(tmp_path)
    collect.collect_youtube(_config(), store)
    rows = list(csv.DictReader((tmp_path / "youtube_windows.csv").open()))
    assert len(rows) == 2
    assert float(rows[0]["median_views"]) == 2000
    assert float(rows[0]["median_views_per_sub"]) == 1.0


def test_reddit_posting_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(self: Any, url: str, params: Any = None, timeout: Any = None):
        if url.endswith("/about.json"):
            return FakeResponse({"data": {"subscribers": 1234, "created_utc": 0}})
        base = 1_700_000_000
        children = [{"data": {"created_utc": base + i * 8640}} for i in range(11)]
        return FakeResponse({"data": {"children": children}})

    monkeypatch.setattr("requests.Session.get", fake_get)
    monkeypatch.setattr(collect.time, "sleep", lambda _s: None)
    store = collect.Store(tmp_path)
    collect.collect_reddit(_config(), store)
    rows = list(csv.DictReader((tmp_path / "reddit.csv").open()))
    assert rows[0]["subscribers"] == "1234"
    assert float(rows[0]["posts_per_day"]) == 10.0  # 10 gaps over 1 day


def _write_fixtures(raw: Path) -> None:
    months = pd.period_range("2016-01", "2026-10", freq="M").astype(str)
    wiki = []
    for niche, growth in (("a", 1.03), ("b", 1.00)):
        for i, month in enumerate(months):
            wiki.append(
                {
                    "niche_id": niche,
                    "article": f"{niche}1",
                    "month": month,
                    "views": int(1000 * growth**i),
                }
            )
    pd.DataFrame(wiki).to_csv(raw / "wiki_monthly.csv", index=False)
    weeks = pd.date_range("2021-11-07", periods=260, freq="W")
    trends = [
        {
            "niche_id": n,
            "keyword": f"kw {n}",
            "timeframe": "today 5-y",
            "date": d.date(),
            "value": 20 + (i // 52) * (5 if n == "a" else 0),
            "anchor_value": 50,
            "is_partial": False,
        }
        for n in ("a", "b")
        for i, d in enumerate(weeks)
    ]
    pd.DataFrame(trends).to_csv(raw / "trends_timeseries.csv", index=False)
    yt = [
        {
            "niche_id": n,
            "keyword": f"kw {n}",
            "window": w,
            "total_results_est": c,
            "median_views_per_day": 10,
            "median_views_per_sub": 0.1,
        }
        for n, w, c in (
            ("a", "recent", 100),
            ("a", "prior_year", 100),
            ("b", "recent", 300),
            ("b", "prior_year", 100),
        )
    ]
    pd.DataFrame(yt).to_csv(raw / "youtube_windows.csv", index=False)


def test_score_ranks_growing_low_supply_niche_first(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_fixtures(raw)
    config = tmp_path / "niches.yaml"
    config.write_text("niches:\n  - {id: a, name: A}\n  - {id: b, name: B}\n")
    money = tmp_path / "money.csv"
    money.write_text("niche_id,low_usd,high_usd,source_1\n")
    df, monthly = score.build_scores(raw, money, config)
    assert list(df.niche_id) == ["a", "b"]
    row_b = df.set_index("niche_id").loc["b"]
    assert np.isclose(row_b.supply_growth, 301 / 101)
    assert row_b.money_flag == "pending"
    score.plot_demand(monthly, df.head(5), tmp_path / "d.png")
    score.plot_scatter(df, tmp_path / "s.png")
    assert (tmp_path / "d.png").exists() and (tmp_path / "s.png").exists()
