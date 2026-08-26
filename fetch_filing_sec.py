"""
Fetch real 10-Q and 10-K filings (MD&A section) for one ticker within a date
range via the SEC API (sec-api.io), and save as two date -> text pickles
under data/01_raw/. Rewritten from
data-pipeline/01_SEC_API_10k10q_download.py, restricted to a single ticker
and an explicit date range, since the SEC API key here has a lifetime quota
of 100 calls.

Because that quota is scarce and non-renewable, this script is deliberately
resumable and saves incrementally:
  - each filing's content is written to disk right after it's fetched, so a
    crash partway through does not lose the SEC calls already spent
  - filings already present in the output pickle (by filed date) are
    skipped on a re-run, so retrying after a partial failure never re-spends
    quota on filings that already succeeded

Requires SEC_KEY in .env.

Usage:
    python fetch_filing_sec.py --symbol TSLA --start 2026-01-01 --end 2026-06-30
"""

import os
import html
import pickle
import argparse
import httpx
from pathlib import Path
from datetime import date, datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from puppy.http_retry import request_with_retry

SEARCH_END_POINT = "https://api.sec-api.io"
EXTRACTOR_END_POINT = "https://api.sec-api.io/extractor"
FORM_ITEM = {
    "10-Q": "part1item2",  # MD&A section
    "10-K": "7",  # MD&A section
}


def _filed_at_to_et_date(filed_at: str) -> date:
    dt = datetime.fromisoformat(filed_at.replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo("America/New_York")).date()


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save(path: Path, data: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def find_filings(
    client: httpx.Client, sec_key: str, symbol: str, form_type: str, start: date, end: date
) -> list[dict]:
    """Full-text search index lookup, filtered client-side to [start, end]."""
    query_payload = {
        "query": {"query_string": {"query": f'ticker:{symbol} AND formType:"{form_type}"'}},
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    response = request_with_retry(
        "POST", SEARCH_END_POINT, client=client, params={"token": sec_key}, json=query_payload
    )
    response.raise_for_status()
    filings = []
    for record in response.json().get("filings", []):
        filed_date = _filed_at_to_et_date(record["filedAt"])
        if not (start <= filed_date <= end):
            continue
        for doc in record["documentFormatFiles"]:
            if doc["type"] == form_type:
                filings.append({"document_url": doc["documentUrl"], "filed_date": filed_date})
                break
    return filings


def fetch_filing_content(client: httpx.Client, sec_key: str, document_url: str, item: str) -> str:
    response = request_with_retry(
        "GET",
        EXTRACTOR_END_POINT,
        client=client,
        params={"url": document_url, "item": item, "token": sec_key},
    )
    response.raise_for_status()
    return html.unescape(" ".join(response.text.split()))


def fetch_filings_for_form(
    client: httpx.Client,
    sec_key: str,
    symbol: str,
    form_type: str,
    start: date,
    end: date,
    out_path: Path,
) -> dict:
    filing_by_date = load_existing(out_path)
    filings = find_filings(client, sec_key, symbol, form_type, start, end)
    print(f"  found {len(filings)} {form_type} filing(s) in range")

    failures = []
    for filing in filings:
        if filing["filed_date"] in filing_by_date:
            print(f"  {filing['filed_date']}: already fetched, skipping")
            continue
        try:
            content = fetch_filing_content(
                client, sec_key, filing["document_url"], FORM_ITEM[form_type]
            )
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            print(f"  {filing['filed_date']}: FAILED ({exc}), will retry on next run")
            failures.append(filing["filed_date"])
            continue
        filing_by_date[filing["filed_date"]] = content
        save(out_path, filing_by_date)  # persist immediately, quota already spent
        print(f"  {filing['filed_date']}: {len(content)} chars, saved to {out_path}")

    if failures:
        print(f"  WARNING: {len(failures)} {form_type} filing(s) failed: {failures}")
    return filing_by_date


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out-dir",
        default="data/01_raw",
        help="Output directory for filing_q_<symbol>.pkl and filing_k_<symbol>.pkl",
    )
    args = parser.parse_args()

    sec_key = os.environ.get("SEC_KEY")
    if not sec_key:
        raise SystemExit("SEC_KEY not set in .env")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / f"filing_q_{args.symbol}.pkl"
    k_path = out_dir / f"filing_k_{args.symbol}.pkl"

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    with httpx.Client(timeout=60.0) as client:
        filing_q = fetch_filings_for_form(client, sec_key, args.symbol, "10-Q", start, end, q_path)
        filing_k = fetch_filings_for_form(client, sec_key, args.symbol, "10-K", start, end, k_path)

    print(f"{args.symbol}: {len(filing_q)} 10-Q -> {q_path}, {len(filing_k)} 10-K -> {k_path}")


if __name__ == "__main__":
    main()
