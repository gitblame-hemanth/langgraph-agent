"""Field and data extraction tool using LLM and regex patterns."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Dates: ISO, US, European, and verbose formats
_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # 2024-01-15
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # 01/15/2024 or 1/15/24
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),  # 01-15-2024
    re.compile(
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
        re.IGNORECASE,
    ),  # 15 January 2024
    re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),  # January 15, 2024
]

# Money amounts: $1,234.56  USD 1234.56  1,234.56 USD  EUR 99.00
_AMOUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[\$\u00A3\u20AC]\s*[\d,]+\.?\d*"),  # $1,234.56 or €99
    re.compile(r"\b(?:USD|EUR|GBP|CAD|AUD)\s*[\d,]+\.?\d*", re.IGNORECASE),  # USD 1234.56
    re.compile(r"[\d,]+\.?\d*\s*(?:USD|EUR|GBP|CAD|AUD)\b", re.IGNORECASE),  # 1234.56 USD
]


def _strip_amount(raw: str) -> float:
    """Parse a raw money string into a float."""
    cleaned = re.sub(r"[^\d.]", "", raw)
    return float(cleaned) if cleaned else 0.0


class ExtractorTool:
    """Extract structured data from document text using LLM and regex."""

    @staticmethod
    async def extract_fields(
        text: str,
        fields: Sequence[str],
        llm: BaseChatModel,
    ) -> dict[str, Any]:
        """Use the LLM to extract named fields from document text.

        Args:
            text: Document text to extract from.
            fields: Field names to extract (e.g. ["invoice_number", "total"]).
            llm: A LangChain chat model instance.

        Returns:
            Dict mapping field names to extracted values. Missing fields
            will have None values.
        """
        if not text or not fields:
            logger.warning("extract_fields.empty_input")
            return dict.fromkeys(fields)

        fields_desc = ", ".join(fields)
        prompt = (
            "Extract the following fields from this document. "
            f"Fields: {fields_desc}\n\n"
            "Return ONLY a JSON object with these exact field names as keys. "
            "Use null for any field you cannot find.\n\n"
            f"Document:\n{text[:6000]}"
        )

        logger.info("extract_fields.start", field_count=len(fields))
        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON from the response — handle markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", str(content))
            json_str = json_match.group(1) if json_match else str(content)
            parsed = json.loads(json_str.strip())

            # Ensure all requested fields are present
            result = {f: parsed.get(f) for f in fields}
            logger.info(
                "extract_fields.done",
                extracted_count=sum(1 for v in result.values() if v is not None),
            )
            return result
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.error("extract_fields.parse_error", error=str(exc))
            return dict.fromkeys(fields)
        except Exception as exc:
            logger.error("extract_fields.error", error=str(exc))
            return dict.fromkeys(fields)

    @staticmethod
    async def extract_table(
        text: str,
        columns: Sequence[str],
        llm: BaseChatModel,
    ) -> list[dict[str, Any]]:
        """Use the LLM to extract tabular data from document text.

        Args:
            text: Document text containing tabular data.
            columns: Expected column names.
            llm: A LangChain chat model instance.

        Returns:
            List of dicts, each representing a row with column-name keys.
        """
        if not text or not columns:
            logger.warning("extract_table.empty_input")
            return []

        cols_desc = ", ".join(columns)
        prompt = (
            "Extract tabular data from this document. "
            f"Expected columns: {cols_desc}\n\n"
            "Return ONLY a JSON array of objects, each with the column names as keys. "
            'Example: [{"col1": "val", "col2": 123}]\n\n'
            f"Document:\n{text[:6000]}"
        )

        logger.info("extract_table.start", column_count=len(columns))
        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", str(content))
            json_str = json_match.group(1) if json_match else str(content)
            parsed = json.loads(json_str.strip())

            if not isinstance(parsed, list):
                logger.warning("extract_table.not_list")
                return []

            # Normalize rows to contain only requested columns
            rows = [{c: row.get(c) for c in columns} for row in parsed if isinstance(row, dict)]
            logger.info("extract_table.done", row_count=len(rows))
            return rows
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.error("extract_table.parse_error", error=str(exc))
            return []
        except Exception as exc:
            logger.error("extract_table.error", error=str(exc))
            return []

    @staticmethod
    def extract_dates(text: str) -> list[str]:
        """Extract date strings from text using regex patterns.

        Args:
            text: Document text.

        Returns:
            Deduplicated list of date strings found in the text.
        """
        if not text:
            return []

        seen: set[str] = set()
        results: list[str] = []
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group().strip()
                if value not in seen:
                    seen.add(value)
                    results.append(value)

        logger.info("extract_dates.done", count=len(results))
        return results

    @staticmethod
    def extract_amounts(text: str) -> list[float]:
        """Extract monetary amounts from text using regex patterns.

        Args:
            text: Document text.

        Returns:
            Deduplicated list of float amounts found in the text.
        """
        if not text:
            return []

        seen: set[float] = set()
        results: list[float] = []
        for pattern in _AMOUNT_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    value = _strip_amount(match.group())
                    if value > 0 and value not in seen:
                        seen.add(value)
                        results.append(value)
                except (ValueError, TypeError):
                    continue

        logger.info("extract_amounts.done", count=len(results))
        return results
