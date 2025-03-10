"""Document search tool using TF-IDF / BM25 ranking."""

from __future__ import annotations

import math
import re
from collections import Counter

import structlog

logger = structlog.get_logger(__name__)


class SearchTool:
    """Search within a document using BM25-style ranking over paragraphs."""

    # BM25 tuning parameters
    _K1: float = 1.5
    _B: float = 0.75

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase tokenize, stripping non-alphanumeric chars."""
        return re.findall(r"[a-z0-9]+", text.lower())

    @classmethod
    def _split_paragraphs(cls, text: str) -> list[str]:
        """Split document text into non-empty paragraphs."""
        raw = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in raw if p.strip()]

    @classmethod
    def search_in_document(
        cls,
        document_text: str,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        """Search for paragraphs matching the query using BM25 scoring.

        Args:
            document_text: Full document text.
            query: Search query string.
            top_k: Number of top-matching paragraphs to return.

        Returns:
            Up to *top_k* paragraphs ranked by relevance, best first.
        """
        if not document_text or not query:
            logger.warning("search.empty_input", has_text=bool(document_text), has_query=bool(query))
            return []

        paragraphs = cls._split_paragraphs(document_text)
        if not paragraphs:
            return []

        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return paragraphs[:top_k]

        # Pre-compute corpus statistics
        doc_tokens: list[list[str]] = [cls._tokenize(p) for p in paragraphs]
        avg_dl = sum(len(d) for d in doc_tokens) / len(doc_tokens) if doc_tokens else 1.0
        n_docs = len(paragraphs)

        # Document frequency for each query term
        df: dict[str, int] = {}
        for term in set(query_tokens):
            df[term] = sum(1 for dt in doc_tokens if term in dt)

        # BM25 score per paragraph
        scores: list[tuple[float, int]] = []
        for idx, tokens in enumerate(doc_tokens):
            tf_map = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in query_tokens:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log((n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
                numerator = tf * (cls._K1 + 1)
                denominator = tf + cls._K1 * (1 - cls._B + cls._B * dl / avg_dl)
                score += idf * numerator / denominator
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [paragraphs[idx] for score, idx in scores[:top_k] if score > 0]

        logger.info("search.done", query=query, result_count=len(results))
        return results if results else paragraphs[:top_k]
