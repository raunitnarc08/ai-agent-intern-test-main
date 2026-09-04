"""Generic conflict detection for relevant, active knowledge-base sections.

Precedence is handled before this module is called. This module deliberately
avoids product names, document names, policy numbers, or evaluation-case logic.
It only flags contradictions when the conflicting statements are both relevant
to the customer's question and share enough topical vocabulary to plausibly
refer to the same subject.
"""

from typing import List, Optional, Set, Tuple, Union
import re

from retrieval.parser import DocumentSection


# Generic mutually-exclusive language patterns. These are intentionally broad,
# but they are only evaluated at the sentence level and must also share topic
# terms before a conflict is reported.
_CONTRADICTION_PATTERNS = (
    (r"\bhand[- ]wash(?:ed|ing)?\b", r"\bdishwasher[- ]safe\b"),
    (r"\bnot\s+refundable\b", r"\brefundable\b"),
    (r"\bnon[- ]refundable\b", r"\brefundable\b"),
    (r"\bnot\s+returnable\b", r"\breturnable\b"),
    (r"\bcannot\s+be\s+returned\b", r"\bcan\s+be\s+returned\b"),
    (r"\bnot\s+covered\b", r"\bcovered\b"),
    (r"\bnot\s+prepaid\b", r"\bprepaid\b"),
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "may", "me", "my", "of", "on", "or", "our", "please",
    "should", "that", "the", "their", "there", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "would",
    "you", "your",
}

# Very generic words do not establish that two passages discuss the same topic.
_GENERIC_TERMS = {
    "policy", "policies", "customer", "customers", "item", "items", "order",
    "orders", "information", "request", "requests", "support", "service",
    "product", "products", "available", "return", "returns", "returned",
    "refund", "refundable", "shipping", "delivery", "days", "day", "safe",
    "care", "please", "must", "should", "can", "cannot", "not",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _terms(text: str) -> Set[str]:
    tokens = re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", _normalize(text))
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in _STOPWORDS
    }


def _topic_terms(text: str) -> Set[str]:
    return _terms(text) - _GENERIC_TERMS


def _query_terms(query: str) -> Set[str]:
    return _terms(query)


def _relevance_score(section: DocumentSection, query_terms: Set[str]) -> int:
    """Score section relevance using title/heading overlap more strongly."""

    title_terms = _terms(section.doc_title)
    heading_terms = _terms(section.heading)
    content_terms = _terms(section.content)

    score = 0
    score += 4 * len(query_terms & title_terms)
    score += 3 * len(query_terms & heading_terms)
    score += len(query_terms & content_terms)
    return score


def _relevant(section: DocumentSection, query_terms: Set[str]) -> bool:
    """Require meaningful overlap rather than two arbitrary common words."""

    if not query_terms:
        return True

    title_terms = _terms(section.doc_title)
    heading_terms = _terms(section.heading)
    content_terms = _terms(section.content)

    title_heading_overlap = query_terms & (title_terms | heading_terms)
    content_overlap = query_terms & content_terms

    # A named/topic term in the title or heading is strong evidence. Otherwise,
    # require at least two query terms in the passage itself.
    if title_heading_overlap:
        return True

    return len(content_overlap) >= min(2, len(query_terms))


def _sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _find_contradiction(
    first: Union[DocumentSection, str],
    second: Union[DocumentSection, str],
) -> Optional[Tuple[str, str]]:
    """Find a contradiction only when the statements share a topic."""

    text_a = first.content if isinstance(first, DocumentSection) else str(first)
    text_b = second.content if isinstance(second, DocumentSection) else str(second)

    context_topics_a = (
        _topic_terms(f"{first.doc_title} {first.heading}")
        if isinstance(first, DocumentSection)
        else set()
    )
    context_topics_b = (
        _topic_terms(f"{second.doc_title} {second.heading}")
        if isinstance(second, DocumentSection)
        else set()
    )

    sentences_a = _sentences(text_a)
    sentences_b = _sentences(text_b)

    for sentence_a in sentences_a:
        normalized_a = _normalize(sentence_a)
        topics_a = _topic_terms(sentence_a) | context_topics_a
        if not topics_a:
            continue

        for sentence_b in sentences_b:
            normalized_b = _normalize(sentence_b)
            topics_b = _topic_terms(sentence_b) | context_topics_b

            # Two shared non-generic terms are a strong signal that the
            # contradictory statements refer to the same subject. For very
            # short statements, one distinctive shared term is sufficient.
            shared_topics = topics_a & topics_b
            if len(shared_topics) < 2 and not (
                len(topics_a) <= 2
                and len(topics_b) <= 2
                and len(shared_topics) >= 1
            ):
                continue

            for first_pat, second_pat in _CONTRADICTION_PATTERNS:
                if re.search(first_pat, normalized_a) and re.search(second_pat, normalized_b):
                    return sentence_a, sentence_b
                if re.search(second_pat, normalized_a) and re.search(first_pat, normalized_b):
                    return sentence_a, sentence_b

    return None


def _excerpt(text: str, limit: int = 280) -> str:
    excerpt = re.sub(r"\s+", " ", text).strip()
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 3].rstrip() + "..."


def detect_conflicts(
    retrieved_sections: List[DocumentSection],
    query: str,
) -> Tuple[bool, str]:
    """Detect likely contradictions among relevant authoritative sections."""

    query_terms = _query_terms(query)

    relevant_sections = [
        section
        for section in retrieved_sections
        if _relevant(section, query_terms)
    ]

    # Prefer stronger query matches first, but retain all relevant sections so
    # an authoritative secondary source can still reveal a genuine conflict.
    relevant_sections.sort(
        key=lambda section: _relevance_score(section, query_terms),
        reverse=True,
    )

    for i, first in enumerate(relevant_sections):
        for second in relevant_sections[i + 1 :]:
            if first.filename == second.filename:
                continue

            contradiction = _find_contradiction(
                first,
                second,
            )
            if contradiction is None:
                continue

            first_statement, second_statement = contradiction

            notice = (
                "The current official sources contain a conflict for this "
                "topic. "
                f"[{first.citation}] states: {first_statement} "
                f"[{second.citation}] states: {second_statement} "
                "Because the authoritative evidence is inconsistent, human "
                "confirmation is required before choosing one instruction "
                "over the other; use the safer interim guidance where "
                "appropriate."
            )
            return True, notice

    return False, ""
