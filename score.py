"""Score niches from the raw CSVs written by ``collect.py``.

Formula (every input is kept as a raw column in ``data/scores.csv``)::

    demand_growth  G_d = geomean(wiki_yoy, trends_yoy)
                     wiki_yoy   = Wikipedia views, last 12 complete months
                                  / previous 12 months
                     trends_yoy = Google Trends, mean of last 52 weeks
                                  / mean of previous 52 weeks
                                  (geomean across the niche's keywords)
    supply_growth  G_s = geomean over keywords of
                         (videos_recent + 1) / (videos_prior_year + 1)
                         (YouTube search.list totalResults, matched windows)
    demand_size    P_d = percentile rank (0-1] of the mean z-score of
                         log(wiki monthly views), log(Trends level vs anchor),
                         log(Reddit subscribers)
    money          M   = niche mid RPM / median mid RPM of niches with data
                         (1.0 and flagged when no cited estimate exists)

    opportunity = (G_d / G_s) * sqrt(P_d) * M

The square root damps small niches: high growth on tiny demand is not an
opportunity, but it should not be ranked out entirely either.

Usage::

    python score.py
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

LOG = logging.getLogger("niche-scout.score")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame if the file is missing."""
    if not path.exists():
        LOG.warning("Missing input %s; dependent metrics will be blank", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def geomean(values: Sequence[float]) -> float:
    """Geometric mean of positive finite values; NaN if none."""
    clean = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if not clean:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in clean) / len(clean)))


def log_slope_annual(series: pd.Series) -> float:
    """Annualised growth from an OLS fit of ln(values) on month index."""
    clean = series[series > 0]
    if len(clean) < 12:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    beta = np.polyfit(x, np.log(clean.to_numpy(dtype=float)), 1)[0]
    return float(math.exp(12 * beta) - 1)


# --------------------------------------------------------------------------- #
# Per-source metrics
# --------------------------------------------------------------------------- #


def wiki_metrics(wiki: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-niche Wikipedia metrics and the monthly niche totals."""
    if wiki.empty:
        return pd.DataFrame(), pd.DataFrame()
    monthly = pd.DataFrame(
        wiki.groupby(["niche_id", "month"], as_index=False)["views"].sum()
    ).sort_values(["niche_id", "month"])
    rows = []
    for niche, grp in monthly.groupby("niche_id"):
        views = grp.set_index("month")["views"]
        last12, prev12 = views.iloc[-12:], views.iloc[-24:-12]
        rows.append(
            {
                "niche_id": niche,
                "wiki_articles": wiki.loc[wiki.niche_id == niche, "article"].nunique(),
                "wiki_last_month": views.index[-1],
                "wiki_monthly_median_12m": float(last12.median()),
                "wiki_yoy": (
                    float(last12.sum() / prev12.sum())
                    if len(prev12) == 12 and prev12.sum() > 0
                    else float("nan")
                ),
                "wiki_cagr_36m": log_slope_annual(views.iloc[-36:]),
                "wiki_cagr_full": log_slope_annual(views),
            }
        )
    return pd.DataFrame(rows), monthly


def trends_metrics(trends: pd.DataFrame) -> pd.DataFrame:
    """Return per-niche Google Trends growth and anchor-relative level."""
    if trends.empty:
        return pd.DataFrame()
    five = trends[(trends.timeframe == "today 5-y") & (~trends.is_partial)]
    rows = []
    for (niche, kw), grp in five.groupby(["niche_id", "keyword"]):
        grp = grp.sort_values("date")
        last, prev = grp.value.iloc[-52:], grp.value.iloc[-104:-52]
        anchor_mean = grp.anchor_value.iloc[-52:].mean()
        rows.append(
            {
                "niche_id": niche,
                "keyword": kw,
                "yoy": last.mean() / prev.mean() if prev.mean() > 0 else np.nan,
                "level_vs_anchor": (
                    last.mean() / anchor_mean if anchor_mean > 0 else np.nan
                ),
                "low_resolution": bool(grp.value.max() < 5),
            }
        )
    per_kw = pd.DataFrame(rows)
    if per_kw.empty:
        return per_kw
    return (
        per_kw.groupby("niche_id")
        .agg(
            trends_yoy=("yoy", lambda s: geomean(list(s))),
            trends_level=("level_vs_anchor", "mean"),
            trends_low_res_kw=("low_resolution", "sum"),
        )
        .reset_index()
    )


def youtube_metrics(yt: pd.DataFrame) -> pd.DataFrame:
    """Return per-niche YouTube supply growth and audience-per-video stats."""
    if yt.empty:
        return pd.DataFrame()
    wide = yt.pivot_table(
        index=["niche_id", "keyword"],
        columns="window",
        values=["total_results_est", "median_views_per_day", "median_views_per_sub"],
        aggfunc="first",
    )
    wide.columns = ["_".join(map(str, c)) for c in wide.columns]
    wide = wide.reset_index()
    needed = {"total_results_est_recent", "total_results_est_prior_year"}
    if not needed.issubset(wide.columns):
        return pd.DataFrame()
    wide["supply_ratio"] = (wide.total_results_est_recent + 1) / (
        wide.total_results_est_prior_year + 1
    )
    agg = wide.groupby("niche_id").agg(
        yt_videos_recent=("total_results_est_recent", "sum"),
        yt_videos_prior_year=("total_results_est_prior_year", "sum"),
        supply_growth=("supply_ratio", lambda s: geomean(list(s))),
    )
    for col in ("median_views_per_day_recent", "median_views_per_sub_recent"):
        if col in wide.columns:
            agg[f"yt_{col}"] = wide.groupby("niche_id")[col].median()
    return agg.reset_index()


def reddit_metrics(reddit: pd.DataFrame) -> pd.DataFrame:
    """Return per-niche Reddit audience size and posting rate."""
    if reddit.empty:
        return pd.DataFrame()
    return (
        reddit.groupby("niche_id")
        .agg(
            reddit_subscribers=("subscribers", "sum"),
            reddit_posts_per_day=("posts_per_day", "sum"),
            reddit_subs_ok=("subreddit", "count"),
        )
        .reset_index()
    )


def money_metrics(money: pd.DataFrame) -> pd.DataFrame:
    """Return per-niche money weight from cited RPM/CPM estimates."""
    if money.empty or money[["low_usd", "high_usd"]].isna().all().all():
        return pd.DataFrame()
    money = money.dropna(subset=["low_usd", "high_usd"]).copy()
    money["mid"] = (money.low_usd + money.high_usd) / 2
    per = money.groupby("niche_id").agg(
        money_low_usd=("low_usd", "min"),
        money_high_usd=("high_usd", "max"),
        money_mid_usd=("mid", "median"),
        money_sources=("source_1", "count"),
    )
    per["money_weight"] = per.money_mid_usd / per.money_mid_usd.median()
    return per.reset_index()


# --------------------------------------------------------------------------- #
# Scoring and output
# --------------------------------------------------------------------------- #


def zscore(series: pd.Series) -> pd.Series:
    """Standard score ignoring NaN; zeros if there is no spread."""
    std = series.std(ddof=0)
    if not std or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def build_scores(
    raw: Path, money_path: Path, config: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all metrics and compute the opportunity score."""
    names = {
        n["id"]: n["name"]
        for n in yaml.safe_load(config.read_text(encoding="utf-8"))["niches"]
    }
    base = pd.DataFrame({"niche_id": list(names), "niche": list(names.values())})
    wiki, monthly = wiki_metrics(read_csv(raw / "wiki_monthly.csv"))
    parts = [
        wiki,
        trends_metrics(read_csv(raw / "trends_timeseries.csv")),
        youtube_metrics(read_csv(raw / "youtube_windows.csv")),
        reddit_metrics(read_csv(raw / "reddit.csv")),
        money_metrics(read_csv(money_path)),
    ]
    df = base
    for part in parts:
        if not part.empty:
            df = df.merge(part, on="niche_id", how="left")

    def col(name: str) -> pd.Series:
        return df[name] if name in df else pd.Series(np.nan, index=df.index)

    df["demand_growth"] = [
        geomean([w, t]) for w, t in zip(col("wiki_yoy"), col("trends_yoy"), strict=True)
    ]
    size_parts = [
        zscore(np.log(col(c).where(col(c) > 0)))
        for c in ("wiki_monthly_median_12m", "trends_level", "reddit_subscribers")
    ]
    df["demand_size_z"] = pd.concat(size_parts, axis=1).mean(axis=1)
    df["demand_size_pct"] = df.demand_size_z.rank(pct=True)
    df["supply_growth"] = col("supply_growth")
    df["money_weight"] = col("money_weight")
    df["money_flag"] = np.where(df.money_weight.isna(), "pending", "cited")
    weight = df.money_weight.fillna(1.0)
    df["opportunity"] = (
        df.demand_growth / df.supply_growth * np.sqrt(df.demand_size_pct) * weight
    )
    df["rank"] = df.opportunity.rank(ascending=False, method="min")
    return df.sort_values("rank"), monthly


def plot_demand(monthly: pd.DataFrame, top: pd.DataFrame, path: Path) -> None:
    """Line chart: 12-month rolling Wikipedia views, indexed to 2016 = 100."""
    if monthly.empty or top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for _, row in top.iterrows():
        series = monthly[monthly.niche_id == row.niche_id].set_index("month").views
        rolling = series.rolling(12).mean().dropna()
        if rolling.empty:
            continue
        indexed = rolling / rolling.iloc[0] * 100
        ax.plot(pd.to_datetime(indexed.index), indexed.values, label=row.niche, lw=2)
    ax.axhline(100, color="#999", lw=0.8, ls="--")
    ax.set_title("Demand trend, top 5 niches by opportunity")
    ax.set_ylabel("Wikipedia views, 12-mo rolling (first window = 100)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Source: Wikimedia Pageviews API (en.wikipedia, user agent)",
        fontsize=7,
        color="#666",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, path: Path) -> None:
    """Scatter: opportunity (log) vs money weight; hollow = money pending."""
    data = df.dropna(subset=["opportunity"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    x = data.money_weight.fillna(1.0)
    cited = data.money_flag == "cited"
    ax.scatter(x[cited], data.opportunity[cited], s=60, color="#2b6cb0", label="cited")
    ax.scatter(
        x[~cited],
        data.opportunity[~cited],
        s=60,
        facecolors="none",
        edgecolors="#2b6cb0",
        label="money pending (=1.0)",
    )
    for _, row in data.iterrows():
        ax.annotate(
            row.niche_id,
            (row.money_weight if pd.notna(row.money_weight) else 1.0, row.opportunity),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Money weight (mid RPM / median mid RPM)")
    ax.set_ylabel("Opportunity score (log)")
    ax.set_title("Opportunity vs money")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Compute scores, write the ranked table and charts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Score niches from raw CSVs")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--money", type=Path, default=Path("data/money_estimates.csv"))
    parser.add_argument("--config", type=Path, default=Path("niches.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--charts", type=Path, default=Path("charts"))
    args = parser.parse_args(argv)

    df, monthly = build_scores(args.raw, args.money, args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    args.charts.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "scores.csv", index=False, float_format="%.4g")
    summary_cols = [
        c
        for c in (
            "rank",
            "niche",
            "opportunity",
            "demand_growth",
            "supply_growth",
            "demand_size_pct",
            "money_weight",
            "money_flag",
            "wiki_monthly_median_12m",
            "wiki_yoy",
            "trends_yoy",
            "yt_videos_recent",
            "yt_videos_prior_year",
            "reddit_subscribers",
        )
        if c in df
    ]
    (args.out / "scores.md").write_text(
        df[summary_cols].to_markdown(index=False, floatfmt=".3g"), encoding="utf-8"
    )
    plot_demand(
        monthly,
        df.dropna(subset=["opportunity"]).head(5),
        args.charts / "demand_top5.png",
    )
    plot_scatter(df, args.charts / "opportunity_vs_money.png")
    LOG.info("Scored %d niches", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
