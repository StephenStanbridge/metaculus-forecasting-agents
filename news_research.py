"""
AskNews latest-news search, used as a third research source next to the two
web-search LLMs. One call per question (the "latest news" strategy costs one
AskNews call; archive searches cost more, so they are not used).

Reads the API key from ASKNEWS_SECRET (AskNews issues a single API key).
Returns "" when the key is missing or anything fails, so research never
breaks because of the news source.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

NEWS_ARTICLES = 10
NEWS_HOURS_BACK = 48
NEWS_TIMEOUT_S = 60
MAX_CHARS = 12000


def asknews_key() -> str | None:
    key = (os.getenv("ASKNEWS_SECRET") or "").strip()
    return key or None


async def latest_news(query: str) -> str:
    key = asknews_key()
    query = (query or "").strip()
    if not key or not query:
        return ""
    try:
        from asknews_sdk import AsyncAskNewsSDK

        async with AsyncAskNewsSDK(api_key=key, timeout=NEWS_TIMEOUT_S) as sdk:
            response = await sdk.news.search_news(
                query=query[:400],
                n_articles=NEWS_ARTICLES,
                return_type="string",
                method="nl",
                strategy="latest news",
                hours_back=NEWS_HOURS_BACK,
                historical=False,
            )
        text = (response.as_string or "").strip()
        logger.info(f"AskNews returned {len(text)} chars for query {query[:80]!r}")
        return text[:MAX_CHARS]
    except Exception as exc:  # the news source must never break research
        logger.warning(f"AskNews search failed: {type(exc).__name__}: {exc}")
        return ""
