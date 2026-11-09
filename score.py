"""Score niches from the raw CSVs written by ``collect.py``.

Formula (every input is kept as a raw column in ``data/scores.csv``)::

    opportunity = (D / C) * sqrt(P) * M

    D  demand growth vs the median niche
         = geomean(wiki_yoy / median wiki_yoy, trends_yoy / median trends_yoy)
         wiki_yoy   = Wikipedia views, last 12 complete months / previous 12
         trends_yoy = Google Trends, mean of last 52 weeks / previous 52,
                      each keyword queried alone (geomean across keywords)
    C  competition pressure vs the median niche
         = geomean(supply_growth / median, median reach / reach), with a
           missing supply_growth counted as typical (1.0)
         supply_growth = YouTube videos, last 14 days / same 14 days a year
                         earlier; counts at the 1,000,000 cap are excluded
         reach         = median views per subscriber of recent videos from
                         channels with >= 1,000 subscribers
    P  demand size: percentile rank of the mean z-score of log Wikipedia
       views, log Trends level vs anchor, log Reddit subscribers
    M  money: niche mid RPM / median mid RPM (1.0 and flagged until cited)

D and C are both 1.0 for a typical niche, so opportunity above sqrt(P) * M
means growing faster, or facing less competition, than a typical niche.

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


MIN_BATCH_PEAK = 20  # below this, integer rounding swamps a batched series
MIN_REACH_CHANNELS = 20  # channels behind a keyword's views/subscriber median
YT_RESULTS_CAP = 1_000_000  # search.list never reports more than this
MAX_ZERO_WEEKS = 13  # base year with more zero weeks is too sparse for a ratio


def _yoy(values: pd.Series) -> float:
    """Mean of the last 52 weeks over the previous 52; NaN if undefined."""
    last, prev = values.iloc[-52:], values.iloc[-104:-52]
    if len(prev) < 52 or prev.mean() <= 0:
        return float("nan")
    return float(last.mean() / prev.mean())


def keyword_trends(trends: pd.DataFrame, solo: pd.DataFrame) -> pd.DataFrame:
    """Per-keyword Trends growth and size.

    Growth comes from the keyword's own series (``trends_solo.csv``), where
    the keyword is scaled to its own peak. Without one, the anchored batch
    series is used only if it peaks at ``MIN_BATCH_PEAK`` or more. Size is
    always the anchored level, ``mean(last 52 weeks) / mean(anchor)``.
    """
    rows: list[dict[str, object]] = []
    five = pd.DataFrame()
    if not trends.empty:
        five = trends[(trends.timeframe == "today 5-y") & (~trends.is_partial)]
    solo_ok = pd.DataFrame()
    if not solo.empty:
        solo_ok = solo[~solo.is_partial]
    keys = set()
    for frame in (five, solo_ok):
        if not frame.empty:
            keys |= set(
                map(tuple, frame[["niche_id", "keyword"]].drop_duplicates().values)
            )
    for niche, kw in sorted(keys):
        row: dict[str, object] = {"niche_id": niche, "keyword": kw}
        batch = (
            five[(five.niche_id == niche) & (five.keyword == kw)].sort_values("date")
            if not five.empty
            else pd.DataFrame()
        )
        own = (
            solo_ok[(solo_ok.niche_id == niche) & (solo_ok.keyword == kw)].sort_values(
                "date"
            )
            if not solo_ok.empty
            else pd.DataFrame()
        )
        if not batch.empty:
            anchor = batch.anchor_value.iloc[-52:].mean()
            row["level_vs_anchor"] = (
                batch.value.iloc[-52:].mean() / anchor if anchor > 0 else np.nan
            )
            row["batch_peak"] = int(batch.value.max())
        sparse = (
            not own.empty
            and int((own.value.iloc[-104:-52] == 0).sum()) > MAX_ZERO_WEEKS
        )
        if sparse:
            row["trends_yoy"] = np.nan
            row["mean_last_52w"] = float(own.value.iloc[-52:].mean())
            row["mean_prev_52w"] = float(own.value.iloc[-104:-52].mean())
            row["yoy_from"] = "base year too sparse"
        elif not own.empty:
            row["trends_yoy"] = _yoy(own.value)
            row["mean_last_52w"] = float(own.value.iloc[-52:].mean())
            row["mean_prev_52w"] = float(own.value.iloc[-104:-52].mean())
            row["yoy_from"] = "own series"
        elif not batch.empty and batch.value.max() >= MIN_BATCH_PEAK:
            row["trends_yoy"] = _yoy(batch.value)
            row["mean_last_52w"] = float(batch.value.iloc[-52:].mean())
            row["mean_prev_52w"] = float(batch.value.iloc[-104:-52].mean())
            row["yoy_from"] = "anchored batch"
        else:
            row["trends_yoy"] = np.nan
            row["yoy_from"] = "too little volume"
        rows.append(row)
    return pd.DataFrame(rows)


def trends_metrics(trends: pd.DataFrame, solo: pd.DataFrame) -> pd.DataFrame:
    """Return per-niche Google Trends growth and anchor-relative level."""
    per_kw = keyword_trends(trends, solo)
    if per_kw.empty:
        return per_kw
    if "level_vs_anchor" not in per_kw:
        per_kw["level_vs_anchor"] = np.nan
    return (
        per_kw.groupby("niche_id")
        .agg(
            trends_yoy=("trends_yoy", lambda s: geomean(list(s))),
            trends_level=("level_vs_anchor", "mean"),
            trends_kw_used=("trends_yoy", lambda s: int(s.notna().sum())),
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
    # YouTube caps totalResults at 1,000,000: a capped count is a floor, not a
    # measurement, so its ratio is excluded rather than read as "no growth".
    capped = (wide.total_results_est_recent >= YT_RESULTS_CAP) | (
        wide.total_results_est_prior_year >= YT_RESULTS_CAP
    )
    wide["supply_ratio"] = (wide.total_results_est_recent + 1) / (
        wide.total_results_est_prior_year + 1
    )
    wide.loc[capped, "supply_ratio"] = np.nan
    wide["capped"] = capped
    agg = wide.groupby("niche_id").agg(
        yt_videos_recent=("total_results_est_recent", "sum"),
        yt_videos_prior_year=("total_results_est_prior_year", "sum"),
        supply_growth=("supply_ratio", lambda s: geomean(list(s))),
        yt_kw_capped=("capped", "sum"),
    )
    for col in ("median_views_per_day_recent", "median_views_per_sub_recent"):
        if col in wide.columns:
            agg[f"yt_{col}"] = wide.groupby("niche_id")[col].median()
    # Reach ratio (median views / subscribers of channels with >= 1,000 subs),
    # kept only for keywords with enough channels behind the median.
    recent = yt[yt.window == "recent"].copy()
    if {"median_views_per_sub", "n_channels_ratio"}.issubset(recent.columns):
        ok = recent[recent.n_channels_ratio >= MIN_REACH_CHANNELS]
        agg["yt_reach"] = ok.groupby("niche_id").median_views_per_sub.apply(
            lambda s: geomean(list(s))
        )
        agg["yt_reach_kw"] = ok.groupby("niche_id").size()
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
        trends_metrics(
            read_csv(raw / "trends_timeseries.csv"), read_csv(raw / "trends_solo.csv")
        ),
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

    def rel(series: pd.Series) -> pd.Series:
        """Ratio to the median niche: 1.0 = typical, >1 better than typical."""
        return series / series.median()

    # Baseline (Salganik, Bit by Bit): Wikipedia traffic is falling almost
    # everywhere, so each source is compared with the median niche before
    # combining. This also stops a niche missing one source being judged on
    # the other source's general drift.
    df["wiki_yoy_rel"] = rel(col("wiki_yoy"))
    df["trends_yoy_rel"] = rel(col("trends_yoy"))
    df["demand_growth"] = [
        geomean([w, t]) for w, t in zip(col("wiki_yoy"), col("trends_yoy"), strict=True)
    ]
    df["demand_growth_rel"] = [
        geomean([w, t]) for w, t in zip(df.wiki_yoy_rel, df.trends_yoy_rel, strict=True)
    ]
    size_parts = [
        zscore(np.log(col(c).where(col(c) > 0)))
        for c in ("wiki_monthly_median_12m", "trends_level", "reddit_subscribers")
    ]
    df["demand_size_z"] = pd.concat(size_parts, axis=1).mean(axis=1)
    df["demand_size_pct"] = df.demand_size_z.rank(pct=True)

    # Competition pressure C, from two supply signals relative to the median:
    #   supply growth (more new videos than a year ago = more pressure) and
    #   reach (views per subscriber; higher = audience outruns supply = less).
    df["supply_growth"] = col("supply_growth")
    df["supply_growth_rel"] = rel(df.supply_growth)
    df["yt_reach"] = col("yt_reach")
    df["reach_rel"] = rel(df.yt_reach)
    # A missing signal counts as typical (1.0), so each signal always carries
    # the same weight instead of one standing in for both.
    df["competition"] = [
        geomean([g, 1 / r]) if r and r > 0 else np.nan
        for g, r in zip(df.supply_growth_rel.fillna(1.0), df.reach_rel, strict=True)
    ]
    df["competition_basis"] = np.select(
        [
            df.supply_growth_rel.notna() & df.reach_rel.notna(),
            df.reach_rel.notna(),
            df.supply_growth_rel.notna(),
        ],
        ["growth+reach", "reach only", "growth only"],
        default="none",
    )
    df["money_weight"] = col("money_weight")
    df["money_flag"] = np.where(df.money_weight.isna(), "pending", "cited")
    weight = df.money_weight.fillna(1.0)
    df["opportunity"] = (
        df.demand_growth_rel / df.competition * np.sqrt(df.demand_size_pct) * weight
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
