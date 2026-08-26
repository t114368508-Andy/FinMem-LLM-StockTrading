"""
Fetch real news for one ticker via the Tavily Search API, once per week
(sparse by design, matching the free-tier plan), and save as a
date -> list[str] pickle under data/01_raw/.

Saves after every week (progress tracked in a small `<out>.progress.pkl`
sidecar file, so a week with zero articles is still remembered as done) and
skips weeks already fetched on a re-run, so a mid-run failure never
re-queries weeks that already succeeded.

Requires TAVILY_API_KEY in .env.

Usage:
    python fetch_news_tavily.py --symbol TSLA --start 2026-01-01 --end 2026-06-30
"""

import os
import html
import time
import pickle
import argparse
import httpx
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

from puppy.http_retry import request_with_retry

TAVILY_END_POINT = "https://api.tavily.com/search"


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


def query_week(api_key: str, symbol: str, week_start: date, week_end: date) -> list[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": f"{symbol} stock news",
        "topic": "news",
        "search_depth": "basic",
        "max_results": 5,
        "start_date": week_start.isoformat(),
        "end_date": week_end.isoformat(),
    }
    response = request_with_retry(
        "POST", TAVILY_END_POINT, headers=headers, json=payload, timeout=60.0
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    return [
        html.unescape(f"{r['title']}: {r['content']}") for r in results if r.get("content")
    ]


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
            articles = query_week(api_key, symbol, week_start, week_end)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            print(f"  {week_start} ~ {week_end}: FAILED ({exc}), will retry on next run")
            continue
        if articles:
            news_by_date[week_start] = articles
        done_weeks.add(week_start)
        save_pickle(out_path, news_by_date)
        save_pickle(progress_path, done_weeks)
        print(f"  {week_start} ~ {week_end}: {len(articles)} articles")
        time.sleep(1)  # be gentle with the free-tier rate limit
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
        help="Output pickle path (default: data/01_raw/news_<symbol>.pkl)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise SystemExit("TAVILY_API_KEY not set in .env")

    out_path = Path(args.out) if args.out else Path("data/01_raw") / f"news_{args.symbol}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    news_by_date = fetch_news(api_key, args.symbol, start, end, out_path)

    total_articles = sum(len(v) for v in news_by_date.values())
    print(
        f"{args.symbol}: {len(news_by_date)} weeks / {total_articles} articles saved to {out_path}"
    )


if __name__ == "__main__":
    main()
