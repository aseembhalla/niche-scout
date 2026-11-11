"""Write compact CSVs for the niche-scout dashboard.

Reads ``data/raw/*.csv`` and ``data/scores.csv`` and writes small,
dashboard-ready files to ``dashboard/``. Only aggregation happens here;
every number traces back to a raw row.

Usage::

    python export_dashboard.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from score import keyword_trends

LOG = logging.getLogger("niche-scout.export")


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV or return an empty frame when it is missing."""
    if not path.exists():
        LOG.warning("Missing %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def wiki_niche_monthly(raw: Path) -> pd.DataFrame:
    """Sum article pageviews to one row per niche and month."""
    wiki = _read(raw / "wiki_monthly.csv")
    if wiki.empty:
        return wiki
    monthly = pd.DataFrame(
        wiki.groupby(["niche_id", "month"], as_index=False)["views"].sum()
    )
    return monthly.sort_values(["niche_id", "month"])


def trends_keywords(raw: Path) -> pd.DataFrame:
    """One row per keyword: growth, where it came from, and size vs anchor."""
    frame = keyword_trends(
        _read(raw / "trends_timeseries.csv"), _read(raw / "trends_solo.csv")
    )
    return frame.round(4) if not frame.empty else frame


def trends_rising_top(raw: Path, per_niche: int = 5) -> pd.DataFrame:
    """Top rising related queries per niche."""
    rising = _read(raw / "trends_rising.csv")
    if rising.empty:
        return rising
    rising["rise_num"] = pd.to_numeric(rising.rise, errors="coerce")
    rising = rising.sort_values("rise_num", ascending=False)
    top = rising.groupby("niche_id").head(per_niche)
    return top[["niche_id", "keyword", "rising_query", "rise"]].reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Write every dashboard CSV that has data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export dashboard CSVs")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("docs/data"))
    args = parser.parse_args(argv)
    raw = args.data / "raw"
    args.out.mkdir(parents=True, exist_ok=True)

    frames = {
        "wiki_niche_monthly.csv": wiki_niche_monthly(raw),
        "trends_keywords.csv": trends_keywords(raw),
        "trends_rising_top.csv": trends_rising_top(raw),
    }
    for name, frame in frames.items():
        if frame.empty:
            LOG.warning("No data for %s", name)
            continue
        frame.to_csv(args.out / name, index=False)
        LOG.info("Wrote %s (%d rows)", name, len(frame))

    copies = {
        args.data / "scores.csv": "niche_scores.csv",
        raw / "youtube_windows.csv": "youtube_keywords.csv",
        raw / "reddit.csv": "reddit.csv",
        raw / "youtube_panel_channels.csv": "youtube_panel_channels.csv",
        raw / "youtube_daily.csv": "youtube_daily.csv",
        raw / "errors.csv": "errors.csv",
        args.data / "money_estimates.csv": "money_estimates.csv",
    }
    for src, name in copies.items():
        if src.exists():
            shutil.copyfile(src, args.out / name)
            LOG.info("Copied %s", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
