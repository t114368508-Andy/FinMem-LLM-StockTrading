"""
Merge the three raw pickles produced by fetch_price.py / fetch_news_tavily.py
/ fetch_filing_sec.py into the single date -> OneDateRecord pickle that
puppy/environment.py expects (data/03_model_input/<symbol>_demo.pkl).

Written from scratch instead of reusing data-pipeline/04-data_pipeline.py,
because that script saves each date as a 4-tuple (price, news, filing_q,
filing_k), which does not match the dict schema puppy/environment.py's
OneDateRecord actually validates against.

--news-finnhub is optional: when given, its articles are appended to the
--news (Tavily) articles on any date both sources have, instead of one
replacing the other.

Usage:
    python merge_dataset.py --symbol TSLA \
        --price data/01_raw/price_TSLA.pkl \
        --news data/01_raw/news_TSLA.pkl \
        --news-finnhub data/01_raw/news_finnhub_TSLA.pkl \
        --filing-q data/01_raw/filing_q_TSLA.pkl \
        --filing-k data/01_raw/filing_k_TSLA.pkl
"""

import pickle
import argparse
from pathlib import Path

from puppy.environment import OneDateRecord


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def merge(
    symbol: str,
    price: dict,
    news: dict,
    filing_q: dict,
    filing_k: dict,
    news_finnhub: dict,
) -> dict:
    env_data = {}
    for cur_date, cur_price in price.items():
        combined_news = news.get(cur_date, []) + news_finnhub.get(cur_date, [])
        record = {
            "price": {symbol: cur_price},
            "filing_k": {symbol: filing_k[cur_date]} if cur_date in filing_k else {},
            "filing_q": {symbol: filing_q[cur_date]} if cur_date in filing_q else {},
            "news": {symbol: combined_news} if combined_news else {},
        }
        OneDateRecord.model_validate(record)  # fail fast on schema mismatch
        env_data[cur_date] = record
    return env_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--price", required=True, type=Path)
    parser.add_argument("--news", required=True, type=Path)
    parser.add_argument("--news-finnhub", default=None, type=Path)
    parser.add_argument("--filing-q", required=True, type=Path)
    parser.add_argument("--filing-k", required=True, type=Path)
    parser.add_argument(
        "--out",
        default=None,
        help="Output pickle path (default: data/03_model_input/<symbol>_demo.pkl)",
    )
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else Path("data/03_model_input") / f"{args.symbol.lower()}_demo.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    price = load_pickle(args.price)
    news = load_pickle(args.news) if args.news.exists() else {}
    news_finnhub = (
        load_pickle(args.news_finnhub)
        if args.news_finnhub and args.news_finnhub.exists()
        else {}
    )
    filing_q = load_pickle(args.filing_q) if args.filing_q.exists() else {}
    filing_k = load_pickle(args.filing_k) if args.filing_k.exists() else {}

    env_data = merge(args.symbol, price, news, filing_q, filing_k, news_finnhub)

    with open(out_path, "wb") as f:
        pickle.dump(env_data, f)

    news_days = sum(1 for r in env_data.values() if r["news"])
    filing_q_days = sum(1 for r in env_data.values() if r["filing_q"])
    filing_k_days = sum(1 for r in env_data.values() if r["filing_k"])
    print(
        f"{args.symbol}: {len(env_data)} dates merged -> {out_path} "
        f"({news_days} with news, {filing_q_days} with filing_q, {filing_k_days} with filing_k)"
    )


if __name__ == "__main__":
    main()
