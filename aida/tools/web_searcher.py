"""Wrapper around external web search providers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import requests


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


class WebSearcher:
    """Call a Google Custom Search API compatible endpoint if configured."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.engine_id = os.getenv("GOOGLE_CSE_ID")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.engine_id)

    def search(self, query: str, *, num_results: int = 5) -> List[SearchResult]:
        if not query:
            return []
        if not self.is_configured():
            return [
                SearchResult(
                    title="검색 API 미구성",
                    link="",
                    snippet=(
                        "Google Custom Search API 키 또는 검색 엔진 ID가 설정되지 않았습니다. "
                        "환경 변수 GOOGLE_API_KEY 및 GOOGLE_CSE_ID를 구성해주세요."
                    ),
                )
            ]

        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.engine_id,
            "num": max(1, min(num_results, 10)),
        }

        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1", params=params, timeout=15
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure handling
            return [
                SearchResult(
                    title="검색 요청 실패",
                    link="",
                    snippet=f"외부 검색 중 오류가 발생했습니다: {exc}",
                )
            ]

        payload = response.json()
        items = payload.get("items", [])

        results: List[SearchResult] = []
        for item in items:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
            )
        return results
