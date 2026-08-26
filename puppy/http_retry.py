"""
Shared HTTP retry helper for every external API call in this project
(Gemini chat/embedding, SEC, Tavily, Finnhub). Retries on rate limits (429),
transient server errors (5xx), and network-level failures (timeouts,
connection resets) with exponential backoff. Does NOT retry other 4xx
errors (bad request, auth, not found) since those are not transient and
retrying would just waste API quota without ever succeeding.
"""

import time
import httpx
from typing import Optional

RETRIABLE_STATUS = {429, 500, 502, 503, 504}


def _is_daily_quota_exceeded(response: httpx.Response) -> bool:
    """
    Detects a Gemini 429 caused by a per-day quota ceiling (e.g.
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier"), as opposed to a
    short-lived per-minute rate limit. A day-level quota won't clear for
    hours, so retrying it with backoff (even up to backoff_cap) just wastes
    minutes for nothing -- observed directly in this project: an 8-attempt
    retry cycle against an exhausted free-tier daily quota still failed
    after ~4.5 minutes of waiting. Backoff is only worth doing for the
    genuinely transient case.
    """
    if response.status_code != 429:
        return False
    try:
        violations = response.json()["error"]["details"]
    except (ValueError, KeyError, TypeError):
        return False
    for detail in violations:
        for violation in detail.get("violations", []):
            if "PerDay" in violation.get("quotaId", ""):
                return True
    return False


class MinIntervalPacer:
    """
    Enforces a minimum spacing between successive calls to the same
    downstream service, so a burst of requests (e.g. one day's memory
    embedding immediately followed by its chat decision call) doesn't
    itself trigger a 429 -- instead of only reacting to 429s after they
    already happened, this reduces how often they happen in the first
    place. Gemini chat and embedding calls share one pacer instance since
    they're billed against the same per-project rate limit.
    """

    def __init__(self, min_interval: float = 3.0):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


gemini_pacer = MinIntervalPacer(min_interval=10.0)


def request_with_retry(
    method: str,
    url: str,
    *,
    client: Optional[httpx.Client] = None,
    max_retries: int = 8,
    backoff_cap: float = 90.0,
    **kwargs,
) -> httpx.Response:
    send = client.request if client is not None else httpx.request
    response: Optional[httpx.Response] = None
    last_exc: Optional[httpx.TransportError] = None

    for attempt in range(max_retries):
        try:
            response = send(method, url, **kwargs)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, backoff_cap))
            continue
        last_exc = None
        if response.status_code in RETRIABLE_STATUS:
            if _is_daily_quota_exceeded(response):
                return response  # won't clear for hours; give up now, don't burn time
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, backoff_cap))
            continue
        return response

    if last_exc is not None:
        raise last_exc
    return response  # last attempt's non-retriable-but-still-bad response; caller decides
