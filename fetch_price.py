"""
Fetch real daily closing prices for one ticker via yfinance (free, no API key)
and save them as a date -> price pickle under data/01_raw/.

Usage:
    python fetch_price.py --symbol TSLA --start 2026-02-01 --end 2026-08-01
"""

import time
import pickle
import argparse
from pathlib import Path
from datetime import date

import yfinance as yf


def fetch_price(symbol: str, start: str, end: str, max_retries: int = 6) -> dict:
    """Download daily closing prices and return {date: float}."""
    data = None
    last_exc = None
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            last_exc = None
            break
        except Exception as exc:  # yfinance wraps requests/curl_cffi errors inconsistently
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 60))
    if last_exc is not None:
        raise last_exc
    if data.empty:
        raise ValueError(
            f"No price data returned for {symbol} between {start} and {end}"
        )
    close = data["Close"][symbol]
    price_by_date: dict[date, float] = {}
    for timestamp, value in close.items():
        price_by_date[timestamp.date()] = float(value)
    return price_by_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=None,
        help="Output pickle path (default: data/01_raw/price_<symbol>.pkl)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("data/01_raw") / f"price_{args.symbol}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    price_by_date = fetch_price(args.symbol, args.start, args.end)

    with open(out_path, "wb") as f:
        pickle.dump(price_by_date, f)

    print(f"{args.symbol}: {len(price_by_date)} trading days saved to {out_path}")


if __name__ == "__main__":
    main()
