import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CACHE_DIR,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_BASE_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K_RETRIEVAL,
)
from retrieval.parser import DocumentSection
from retrieval.precedence import load_and_filter_corpus


@dataclass
class SearchResult:
    section: DocumentSection
    score: float

    @property
    def citation(self) -> str:
        return self.section.citation


STOPWORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "is",
    "are",
    "do",
    "does",
    "can",
    "could",
    "would",
    "should",
    "how",
    "what",
    "where",
    "when",
    "why",
    "my",
    "your",
    "our",
    "i",
    "you",
    "me",
    "it",
    "this",
    "that",
    "about",
    "with",
    "from",
    "have",
    "has",
    "had",
    "be",
    "been",
    "was",
    "were",
    "will",
    "just",
    "please",
}


def tokenize(text: str) -> Set[str]:
    """Tokenize text for lightweight lexical relevance."""
    words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower())
    return {
        word
        for word in words
        if word not in STOPWORDS and len(word) > 1
    }


# These aliases improve recall for normal customer language.
#
# Important:
# The aliases deliberately contain terminology/synonyms rather than
# authoritative policy numbers or policy outcomes. The knowledge base remains
# the source of truth for all company-specific facts.
QUERY_ALIASES = {
    # Returns / change of mind
    "send it back": "return returns",
    "send this back": "return returns",
    "send the item back": "return returns",
    "something back": "return returns",
    "how long do i have to return": "return window timeframe deadline",
    "how long can i return": "return window timeframe deadline",
    "when can i return": "return window eligibility",
    "when do i have to return": "return window deadline",
    "send something back": "return window returns eligibility",
    "change my mind": "return returns change of mind",
    "changed my mind": "return returns change of mind",
    "don't want it": "return change of mind",
    "do not want it": "return change of mind",
    # Membership
    "loyalty member": "membership member benefits",
    "loyalty membership": "membership member benefits",
    "member benefits": "membership benefits",
    "membership benefits": "membership benefits",
    "trailplus": "membership benefits",
    # Gift cards
    "store credit": "gift card",
    "giftcard": "gift card",
    "gift card problem": "gift card checkout",
    "gift card error": "gift card checkout",
    # Delivery / shipping
    "delivery exception": "carrier exception shipment delivery problem",
    "shipping exception": "carrier exception shipment delivery problem",
    "package exception": "carrier exception shipment delivery problem support",
    "carrier exception": "carrier delivery problem support review",
    "delivery problem": "carrier delivery problem",
    "shipping problem": "carrier delivery problem",
    "package problem": "carrier delivery problem",
    # Damaged / incorrect items
    "broken zipper": "damaged defective wrong item",
    "broken zip": "damaged defective wrong item",
    "damaged item": "damaged defective wrong item",
    "faulty item": "damaged defective wrong item",
    "wrong item": "incorrect damaged item",
    "incorrect item": "wrong damaged item",
    # Warranty
    "lifetime warranty": "warranty coverage warranty period",
    "covered forever": "warranty coverage warranty period",
    "how long is it covered": "warranty coverage warranty period",
}
def expand_query(query: str) -> str:
    """Expand customer-language synonyms without encoding policy answers."""
    expanded = query
    query_lower = query.lower()
    for phrase, expansion in QUERY_ALIASES.items():
        if phrase in query_lower:
            expanded += " " + expansion
    # Add generic domain vocabulary only. The actual policy values must be
    # discovered from retrieved knowledge-base sections.
    has_damage_language = bool(
        re.search(
            r"\b(damaged|broken|defective|faulty|wrong|incorrect|zipper|defect)\b",
            query_lower,
        )
    )
    has_final_sale_language = bool(
        re.search(
            r"\bfinal[\s-]?sale\b",
            query_lower,
        )
    )
    has_shipping_language = bool(
        re.search(
            r"\b(ship|shipping|shipped|delivery|deliver|destination|international)\b",
            query_lower,
        )
    )
    has_return_language = bool(
        re.search(
            r"\b(return|returns|refund|exchange|change of mind)\b",
            query_lower,
        )
    )
    if has_damage_language:
        expanded += " damaged defective wrong item"
    if has_final_sale_language:
        expanded += " final sale return exception"
    if has_shipping_language:
        expanded += " shipping delivery destination"
    if has_return_language:
        expanded += " return eligibility policy"
    return expanded


class KnowledgeBaseIndex:
    def __init__(
        self,
        kb_dir: Path = KNOWLEDGE_BASE_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.kb_dir = kb_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.sections: List[DocumentSection] = []
        self.embeddings: Optional[np.ndarray] = None

        self._build_or_load_index()

    def _get_cache_path(self) -> Path:
        return (
            CACHE_DIR
            / f"kb_index_v3_{self.model_name.replace('/', '_')}.pkl"
        )

    def _build_or_load_index(self):
        cache_path = self._get_cache_path()

        _all_metadata, sections, _excluded = load_and_filter_corpus(
            self.kb_dir
        )

        self.sections = sections

        corpus_texts = [
            (
                f"Document: {sec.doc_title} ({sec.filename})\n"
                f"Section: {sec.heading}\n"
                f"{sec.content}"
            )
            for sec in self.sections
        ]

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                if cached_data.get("texts") == corpus_texts:
                    self.embeddings = cached_data.get("embeddings")
                    if self.embeddings is not None:
                        return
            except Exception:
                pass

        if not corpus_texts:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        self.embeddings = self.model.encode(
            corpus_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "texts": corpus_texts,
                        "embeddings": self.embeddings,
                    },
                    f,
                )
        except Exception:
            pass

    @staticmethod
    def _return_window_query(query_tokens: Set[str]) -> bool:
        # "window", "long", "timeframe", "deadline" are generic duration
        # words that also apply to shipping ETAs, warranty length, etc.
        # (e.g. "Canada, and how long does it take?" is a shipping question,
        # not a return-policy one). Require co-occurrence with an actual
        # return/refund/exchange context word so this boost only fires for
        # genuine return-window questions.
        has_duration_word = bool(
            query_tokens.intersection(
                {"window", "timeframe", "deadline", "long"}
            )
        )
        has_return_context = bool(
            query_tokens.intersection(
                {"return", "returns", "refund", "exchange", "trailplus", "membership"}
            )
        )
        return has_duration_word and has_return_context

    @staticmethod
    def _shipping_query(query_tokens: Set[str]) -> bool:
        return bool(
            query_tokens.intersection(
                {
                    "ship",
                    "shipping",
                    "shipped",
                    "deliver",
                    "delivery",
                    "destination",
                    "country",
                    "international",
                    "canada",
                    "germany",
                    "europe",
                    "uk",
                    "mexico",
                    "australia",
                }
            )
        )

    @staticmethod
    def _returns_query(query_tokens: Set[str]) -> bool:
        return bool(
            query_tokens.intersection(
                {
                    "return",
                    "returns",
                    "refund",
                    "exchange",
                    "window",
                    "eligible",
                    "eligibility",
                    "final",
                    "sale",
                    "migration",
                    "membership",
                    "trailplus",
                }
            )
        )

    @staticmethod
    def _damaged_query(query_tokens: Set[str]) -> bool:
        return bool(
            query_tokens.intersection(
                {
                    "damaged",
                    "damage",
                    "broken",
                    "defective",
                    "defect",
                    "wrong",
                    "incorrect",
                    "faulty",
                    "zipper",
                }
            )
        )

    @staticmethod
    def _is_shipping_section(section: DocumentSection) -> bool:
        text = (
            f"{section.filename} "
            f"{section.doc_title} "
            f"{section.heading}"
        ).lower()

        return (
            "shipping" in text
            or "delivery" in text
            or "international" in text
            or "domestic" in text
        )

    @staticmethod
    def _is_returns_section(section: DocumentSection) -> bool:
        text = (
            f"{section.filename} "
            f"{section.doc_title} "
            f"{section.heading}"
        ).lower()

        return (
            "return" in text
            or "refund" in text
            or "exchange" in text
            or "final-sale" in section.filename.lower()
        )

    @staticmethod
    def _is_damaged_section(section: DocumentSection) -> bool:
        text = (
            f"{section.filename} "
            f"{section.doc_title} "
            f"{section.heading}"
        ).lower()

        return "damaged" in text or "wrong" in text or "defective" in text

    @staticmethod
    def _gift_card_query(query_tokens: Set[str]) -> bool:
        return bool(
            query_tokens.intersection(
                {
                    "gift",
                    "card",
                    "credit",
                    "voucher",
                    "certificate",
                }
            )
        )

    @staticmethod
    def _is_gift_card_section(section: DocumentSection) -> bool:
        text = (
            f"{section.filename} "
            f"{section.doc_title} "
            f"{section.heading}"
        ).lower()

        return "gift-card" in text or "gift card" in text or "gift cards" in text

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[SearchResult]:
        if not self.sections or self.embeddings is None:
            return []

        search_query = expand_query(query)
        query_tokens = tokenize(search_query)

        # Unsupported material/product-property questions should abstain when
        # the KB contains no evidence for the requested property.
        out_of_scope_terms = {
            "vegan",
            "organic",
            "kosher",
            "halal",
            "hypoallergenic",
            "pfas",
            "bpa-free",
        }

        requested_out_of_scope = query_tokens.intersection(
            out_of_scope_terms
        )

        if requested_out_of_scope:
            corpus_lower = [
                sec.content.lower()
                for sec in self.sections
            ]

            if not any(
                term in content
                for term in requested_out_of_scope
                for content in corpus_lower
            ):
                return []

        query_embedding = self.model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        dense_scores = np.dot(self.embeddings, query_embedding)

        scored_results: List[Tuple[float, int]] = []

        for idx, section in enumerate(self.sections):
            dense_score = float(dense_scores[idx])

            section_text = (
                f"{section.doc_title} "
                f"{section.heading} "
                f"{section.content}"
            ).lower()

            section_tokens = tokenize(section_text)
            title_tokens = tokenize(
                f"{section.doc_title} {section.heading}"
            )

            if query_tokens:
                lexical_overlap = (
                    len(query_tokens.intersection(section_tokens))
                    / len(query_tokens)
                )

                title_overlap = (
                    len(query_tokens.intersection(title_tokens))
                    / len(query_tokens)
                )
            else:
                lexical_overlap = 0.0
                title_overlap = 0.0

            shipping_boost = 0.0
            if self._shipping_query(query_tokens):
                if self._is_shipping_section(section):
                    shipping_boost = 0.22

            returns_boost = 0.0
            if self._returns_query(query_tokens):
                if self._is_returns_section(section):
                    returns_boost = 0.20

            return_window_boost = 0.0
            if self._return_window_query(query_tokens):
                heading_lower = section.heading.lower()
                if "return window" in heading_lower:
                    return_window_boost = 0.24
                elif any(
                    term in heading_lower
                    for term in ("return", "returns", "eligibility", "deadline")
                ):
                    return_window_boost = 0.08

            damaged_boost = 0.0
            if self._damaged_query(query_tokens):
                if self._is_damaged_section(section):
                    damaged_boost = 0.25

            gift_card_boost = 0.0
            if self._gift_card_query(query_tokens):
                if self._is_gift_card_section(section):
                    gift_card_boost = 0.22

            # Small evidence boost when the retrieved section shares multiple
            # important lexical terms with the customer query.
            evidence_boost = 0.0

            if len(query_tokens.intersection(section_tokens)) >= 3:
                evidence_boost = 0.05

            # NOTE: lexical_overlap weight lowered from 0.25 -> 0.15 and the
            # category boosts raised. Previously a single incidental word
            # match (e.g. a customer describing an "unused backpack" hitting
            # the unrelated "Item condition" section's "must be unused,
            # unwashed...") could outrank the section a deliberate, purpose-
            # built boost (return_window_boost, damaged_boost, etc.) exists
            # specifically to surface. These boosts should dominate generic
            # overlap, not compete with it on roughly equal footing.
            hybrid_score = (
                0.50 * dense_score
                + 0.15 * lexical_overlap
                + 0.10 * title_overlap
                + shipping_boost
                + returns_boost
                + return_window_boost
                + damaged_boost
                + gift_card_boost
                + evidence_boost
            )

            scored_results.append((hybrid_score, idx))

        scored_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        candidate_limit = min(
            len(scored_results),
            max(top_k * 3, top_k),
        )

        candidates = [
            (score, idx)
            for score, idx in scored_results[:candidate_limit]
            if score >= threshold
        ]

        if not candidates:
            return []

        # Allow multiple sections from the same document so multi-section
        # policies aren't starved by an arbitrary per-doc limit.
        #
        # Previously hard-capped at MAX_PER_DOC = 2. Confirmed against real
        # KB content that this silently discarded relevant evidence: both
        # 04-damaged-or-wrong-items.md and 06-international-shipping.md have
        # 4 sections each, and cases needing 3-4 of those sections
        # (final-sale-damaged-exception, canada-multiturn) only ever
        # received the top 2, regardless of how many sections actually
        # cleared the relevance threshold for that document.
        #
        # Adaptive cap: allow up to the number of that document's sections
        # that cleared `threshold`, capped at a generous ceiling so one
        # dominant document still can't crowd out every other source.
        doc_candidate_counts: Dict[str, int] = {}
        for score, idx in candidates:
            filename = self.sections[idx].filename
            doc_candidate_counts[filename] = doc_candidate_counts.get(filename, 0) + 1

        ADAPTIVE_CAP_CEILING = 4

        selected: List[SearchResult] = []
        selected_indices: Set[int] = set()
        file_counts: Dict[str, int] = {}

        for score, idx in candidates:
            section = self.sections[idx]
            filename = section.filename
            count = file_counts.get(filename, 0)
            doc_max = min(
                doc_candidate_counts.get(filename, 1),
                ADAPTIVE_CAP_CEILING,
            )

            if count < doc_max:
                selected.append(
                    SearchResult(
                        section=section,
                        score=score,
                    )
                )
                selected_indices.add(idx)
                file_counts[filename] = count + 1

            if len(selected) >= top_k:
                break

        # Second pass: fill remaining slots with strongest remaining sections.
        if len(selected) < top_k:
            for score, idx in candidates:
                if idx in selected_indices:
                    continue

                selected.append(
                    SearchResult(
                        section=self.sections[idx],
                        score=score,
                    )
                )
                selected_indices.add(idx)

                if len(selected) >= top_k:
                    break

        selected.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return selected


_global_index: Optional[KnowledgeBaseIndex] = None


def get_kb_index() -> KnowledgeBaseIndex:
    global _global_index

    if _global_index is None:
        _global_index = KnowledgeBaseIndex()

    return _global_index