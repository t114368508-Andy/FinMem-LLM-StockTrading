"""
Fetch real company news for one ticker via the Finnhub company-news API,
querying week by week, and save as a date -> list[str] pickle under
data/01_raw/.

This is a supplementary source to fetch_news_tavily.py. Finnhub's endpoint
turned out to only return its most recent batch of articles regardless of
how wide the from/to range is (a single call across 6 months returned
articles from the last week only) -- so like fetch_news_tavily.py, this
queries one week at a time, letting each week claim its own slice of
results instead of being crowded out by more recent news.

Saves after every week (progress tracked in a small `<out>.progress.pkl`
sidecar file, so a week with zero articles is still remembered as done) and
skips weeks already fetched on a re-run, so a mid-run failure never
re-queries weeks that already succeeded.

Requires FINNHUB_API_KEY in .env.

Usage:
    python fetch_news_finnhub.py --symbol TSLA --start 2026-01-01 --end 2026-06-30
"""

import os
import html
import time
import pickle
import argparse
import httpx
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from dotenv import load_dotenv

from puppy.http_retry import request_with_retry

END_POINT = "https://finnhub.io/api/v1/company-news"


def week_starts(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=7)


def load_pickle(path: Path, default):
    if not path.exists():
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def query_week(api_key: str, symbol: str, week_start: date, week_end: date) -> dict:
    response = request_with_retry(
        "GET",
        END_POINT,
        params={
            "symbol": symbol,
            "from": week_start.isoformat(),
            "to": week_end.isoformat(),
            "token": api_key,
        },
        timeout=60.0,
    )
    response.raise_for_status()

    by_date: dict[date, list[str]] = {}
    for article in response.json():
        published = datetime.fromtimestamp(article["datetime"], tz=timezone.utc).date()
        if not (week_start <= published <= week_end):
            continue
        text = f"{article.get('headline', '')}: {article.get('summary', '')}".strip(": ")
        if text:
            by_date.setdefault(published, []).append(html.unescape(text))
    return by_date


def fetch_news(api_key: str, symbol: str, start: date, end: date, out_path: Path) -> dict:
    progress_path = out_path.with_suffix(".progress.pkl")
    news_by_date = load_pickle(out_path, {})
    done_weeks = load_pickle(progress_path, set())

    for week_start in week_starts(start, end):
        if week_start in done_weeks:
            print(f"  {week_start}: already fetched, skipping")
            continue
        week_end = min(week_start + timedelta(days=6), end)
        try:
            by_date = query_week(api_key, symbol, week_start, week_end)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            print(f"  {week_start} ~ {week_end}: FAILED ({exc}), will retry on next run")
            continue
        for d, articles in by_date.items():
            news_by_date.setdefault(d, []).extend(articles)
        done_weeks.add(week_start)
        save_pickle(out_path, news_by_date)
        save_pickle(progress_path, done_weeks)
        total = sum(len(v) for v in by_date.values())
        print(f"  {week_start} ~ {week_end}: {total} articles across {len(by_date)} day(s)")
        time.sleep(0.2)  # free tier is 60 calls/minute, stay well under it

    return news_by_date


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=None,
        help="Output pickle path (default: data/01_raw/news_finnhub_<symbol>.pkl)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise SystemExit("FINNHUB_API_KEY not set in .env")

    out_path = (
        Path(args.out) if args.out else Path("data/01_raw") / f"news_finnhub_{args.symbol}.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    news_by_date = fetch_news(api_key, args.symbol, start, end, out_path)

    total_articles = sum(len(v) for v in news_by_date.values())
    print(
        f"{args.symbol}: {len(news_by_date)} days / {total_articles} articles saved to {out_path}"
    )


if __name__ == "__main__":
    main()
