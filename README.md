# niche-scout

Measures 20 candidate faceless content niches with free data, then ranks them
by an opportunity score. Configuration lives in `niches.yaml`.

## Setup (once)

1. Create a GitHub repo (e.g. `aseembhalla/niche-scout`) and push this folder.
2. **Settings → Secrets and variables → Actions → New repository secret**
   - `YOUTUBE_API_KEY`: API key from your Google Cloud project with
     *YouTube Data API v3* enabled.
   - Optional, only if Reddit blocks public JSON from GitHub runners:
     `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` from a free "script" app at
     <https://www.reddit.com/prefs/apps>.
3. **Settings → Actions → General → Workflow permissions →** *Read and write*.

## Runs (on demand only)

Nothing runs on a schedule. Start a run from **Actions → collect-and-score →
Run workflow** and pick sources. Keep each day within the YouTube quota
(10,000 units, reset 08:00 UTC):

| Sources | YouTube units |
|---|---|
| `youtube_daily` — videos actually returned for the last 24 h | ~2,000–4,000 |
| `wiki,trends,trends_solo,youtube` — demand + reach sample | ~4,100 |
| `youtube_panel` — upload histories of each niche's ~20 leading channels | up to ~6,000 |

Why not YouTube search counts: `totalResults` is capped at 1,000,000, ignores
date filters, and older videos fade from search within 20–60 days (Efstratiou
2025, arXiv:2506.04422; Rieder, Padilla & Coromina 2025, arXiv:2506.11727).
Supply is therefore counted by ID from channels' upload lists.

## Run

**Actions → collect-and-score → Run workflow.** Leave the defaults to run all four
sources. Results are committed back to the repo:

| File | Contents |
|---|---|
| `data/raw/wiki_monthly.csv` | Monthly human pageviews per article, with source URL |
| `data/raw/trends_timeseries.csv` | Trends 12-month and 5-year series, with anchor values |
| `data/raw/trends_rising.csv` | Rising related queries (12 months) |
| `data/raw/youtube_windows.csv` | Per keyword and window: video count estimate, median views, views/day, views/subscriber |
| `data/raw/youtube_videos.csv` | Video-level raw rows (for competitor validation) |
| `data/raw/reddit.csv` | Subscribers and posts/day per subreddit |
| `data/raw/errors.csv` | Every failed fetch. Gaps stay blank, never zero |
| `data/scores.csv`, `data/scores.md` | Ranked table with every raw metric |
| `charts/*.png` | Demand trend (top 5), opportunity vs money |

Re-run a single source with e.g. `sources = trends`; other sources' files are
left untouched.

### Local run (Windows, PowerShell)

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:YOUTUBE_API_KEY = "..."
python collect.py --sources wiki,youtube,reddit,trends
python score.py
```

Google Trends often blocks GitHub's cloud IPs. If `errors.csv` shows Trends
429s, run `python collect.py --sources trends` locally and commit
`data/raw/trends_*.csv`.

## Scoring

```
opportunity = (G_d / G_s) * sqrt(P_d) * M
```

- **G_d (demand growth):** geometric mean of Wikipedia year-over-year (last 12
  complete months / previous 12) and Trends year-over-year (last 52 weeks /
  previous 52).
- **G_s (supply growth):** geometric mean over keywords of
  `(videos last 90 days + 1) / (videos same 90 days a year earlier + 1)`.
- **P_d (demand size):** percentile rank of mean z(log Wikipedia views,
  log Trends level vs anchor, log Reddit subscribers).
- **M (money):** niche mid RPM / median mid RPM, from `data/money_estimates.csv`
  (≥2 cited sources per niche). Set to 1.0 and flagged `pending` until filled.

Why this shape, from published methods:

- **Ratio of demand to competition.** This is the core of the Keyword
  Effectiveness Index (volume² / competition) and of volume / (difficulty + 1)
  scoring.
- **High-demand, low-supply "whitespace" quadrant.** This is how Parrot
  Analytics maps content demand against supply.
- **Damping for small demand.** Tools note that with no demand there is no
  opportunity, however low the competition. That is the reason for the
  `sqrt(P_d)` damping.
- **Excluding small channels from views/subscriber.** Channels under 1,000
  subscribers are excluded, because a near-zero baseline inflates the ratio
  (OutlierKit's niche audits).
- **Wikipedia as a demand proxy.** Pageviews correlate strongly with search
  frequency for popular keywords (Yoshida et al., arXiv:1509.02218).

## Known limitations

- **YouTube video counts.** `totalResults` is Google's estimate, not an exact
  count. Treat supply growth as directional.
- **YouTube as a proxy.** YouTube supply stands in for still-image platforms,
  since Instagram and Pinterest have no free public count API.
- **pytrends.** The pytrends repo was archived in April 2025. It still works
  for light use but is prone to HTTP 429 errors.
- **YouTube quota.** One full run uses about 80 `search.list` calls (~8,300
  units) of the 10,000/day quota. Don't run YouTube twice in one day.
- **Reddit.** A single snapshot gives posting rate, not posting growth.

## Development

```
pip install -r requirements-dev.txt pytest
black --check . && flake8 . && mypy --ignore-missing-imports collect.py score.py
pytest -q
```
