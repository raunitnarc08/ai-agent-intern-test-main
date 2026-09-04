from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

from config import GROQ_API_KEY, GROQ_MODEL, TURN_LOGS_PATH
from agent.escalation import should_recommend_handoff
from agent.memory import ChatMessage, get_session_memory
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from retrieval.conflict import detect_conflicts
from retrieval.indexer import SearchResult, expand_query, get_kb_index, tokenize
from tools.order_lookup import get_order_lookup_tool


@dataclass
class AgentResponse:
    response: str
    sources: List[str] = field(default_factory=list)
    handoff: bool = False
    tool_called: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)


def extract_order_id_from_text(text: str) -> Optional[str]:
    """Extract an order ID candidate token from user text.

    Harmless case and trailing punctuation normalization is permitted,
    but malformed formats (such as 'ORD 1007' or 'ORD-107') are preserved as-is
    so that OrderLookupTool strictly validates and rejects them without guessing
    or fuzzy-matching.
    """

    match = re.search(
        r"\bORD[-_ ]?\d+\b",
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    raw_token = match.group(0).strip().rstrip(".,!?:;\"'")
    return raw_token.upper()


def _is_order_status_query(text: str) -> bool:
    """Detect requests to inspect an order/package/shipment."""

    q = text.lower()

    direct_patterns = [
        r"\bwhere is (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bwhere's (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bwhen will (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bwhen should (?:my )?(?:order|package|shipment|parcel)\b",
        r"\btrack (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bcheck (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bstatus (?:of|on|for) (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bupdate (?:on|for) (?:my )?(?:order|package|shipment|parcel)\b",
        r"\bhas (?:my )?(?:order|package|shipment|parcel) shipped\b",
        r"\bhas (?:my )?(?:order|package|shipment|parcel) arrived\b",
        r"\bwhen will it arrive\b",
        r"\bwhere is it\b",
        r"\btrack it\b",
        r"\bcheck the status\b",
    ]

    return any(re.search(pattern, q) for pattern in direct_patterns)


def _history_order_id(
    history: List[ChatMessage],
) -> Optional[str]:
    for message in reversed(history):
        found = extract_order_id_from_text(message.content)
        if found:
            return found

    return None


def _should_reuse_history_order_id(
    message: str,
    history: List[ChatMessage],
) -> bool:
    if not history:
        return False

    q = message.lower()

    follow_up_patterns = [
        r"\bwhen will it arrive\b",
        r"\bwhen should it arrive\b",
        r"\bwhere is it\b",
        r"\bwhere's it\b",
        r"\bwhat is its status\b",
        r"\bwhat's its status\b",
        r"\bwhat is the status\b",
        r"\bwhat's the status\b",
        r"\btrack it\b",
        r"\bcheck it\b",
        r"\bcheck the status\b",
        r"\bcan i cancel it\b",
        r"\bcancel it\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern in follow_up_patterns
    )


class AsterRowAgent:
    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model_name: str = GROQ_MODEL,
        min_seconds_between_llm_calls: float = 2.5,
        max_retry_wait_seconds: float = 10.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY", "")
        )

        self.model_name = (
            model_name
            or os.getenv(
                "GROQ_MODEL",
                "llama-3.3-70b-versatile",
            )
        )

        self.kb_index = get_kb_index()
        self.order_tool = get_order_lookup_tool()
        self.memory = get_session_memory()

        self.client = (
            Groq(api_key=self.api_key, max_retries=0)
            if Groq and self.api_key
            else None
        )

        # Client-side pacing: track when the last LLM call finished so
        # consecutive calls (e.g. across an eval run's many cases in quick
        # succession) don't burst-consume a whole minute's TPM/OTPM budget
        # up front. This belongs in the client layer, not the eval harness,
        # since it's a property of talking to a rate-limited API, not a
        # property of the test suite.
        self._last_llm_call_at: float = 0.0
        self._min_seconds_between_llm_calls: float = min_seconds_between_llm_calls
        self._max_retry_wait_seconds: float = max_retry_wait_seconds

        print(
            f"[Config] min_seconds_between_llm_calls="
            f"{self._min_seconds_between_llm_calls}, "
            f"max_retry_wait_seconds="
            f"{self._max_retry_wait_seconds}"
        )

    def _log_turn(self, log_entry: Dict[str, Any]):
        try:
            with open(
                TURN_LOGS_PATH,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        log_entry,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

    def _remember_and_return(
        self,
        session_id: str,
        user_message: str,
        response_text: str,
        *,
        sources: Optional[List[str]] = None,
        handoff: bool = False,
        tool_called: Optional[str] = None,
        tool_arguments: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:

        sources = sources or []
        debug_info = debug_info or {}

        self.memory.add_user_message(
            session_id,
            user_message,
        )
        self.memory.add_assistant_message(
            session_id,
            response_text,
        )

        return AgentResponse(
            response=response_text,
            sources=sources,
            handoff=handoff,
            tool_called=tool_called,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            debug_info=debug_info,
        )

    @staticmethod
    def _collect_sources(
        search_results: List[SearchResult],
        is_conflict: bool,
    ) -> List[str]:

        sources: List[str] = []
        seen = set()

        for result in search_results:
            filename = result.section.filename

            if filename == "13-support-escalation.md":
                continue

            if filename not in seen:
                seen.add(filename)
                sources.append(filename)

        return sources

    @staticmethod
    def _redact_sensitive_values(
        response_text: str,
        tool_result: Optional[Dict[str, Any]],
    ) -> str:
        if not tool_result:
            return response_text

        customer = tool_result.get("customer") or {}
        internal = tool_result.get("internal") or {}

        sensitive_values = [
            customer.get("email"),
            customer.get("shipping_address"),
            customer.get("name"),
            internal.get("warehouse_note"),
            internal.get("risk_score"),
        ]

        for value in sensitive_values:
            if value is None:
                continue

            value_text = str(value).strip()

            if not value_text:
                continue

            if value_text.lower() in response_text.lower():
                return (
                    "I cannot disclose internal customer data, personal "
                    "addresses, emails, or internal operational notes. "
                    "Please contact customer support if you need account "
                    "assistance."
                )

        return response_text

    @staticmethod
    def _guard_gift_card_code_request(
        response_text: str,
        user_message: str,
    ) -> str:
        q = user_message.lower()

        if not re.search(
            r"\b(gift[\s-]?card|store credit)\b",
            q,
        ):
            return response_text

        dangerous_patterns = [
            r"share your (full )?gift[- ]?card code",
            r"provide your (full )?gift[- ]?card code",
            r"enter your (full )?gift[- ]?card code",
            r"send your (full )?gift[- ]?card code",
            r"give me your (full )?gift[- ]?card code",
        ]

        if any(
            re.search(pattern, response_text, re.IGNORECASE)
            for pattern in dangerous_patterns
        ):
            return (
                "Please contact customer support without sharing your "
                "full gift-card code in chat."
            )

        return response_text

    def chat(
        self,
        session_id: str,
        user_message: str,
    ) -> AgentResponse:

        user_message_clean = user_message.strip()

        if not user_message_clean:
            response_text = (
                "Please tell me what you need help with."
            )

            response = self._remember_and_return(
                session_id,
                user_message_clean,
                response_text,
            )

            self._log_turn(
                {
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "session_id": session_id,
                    "user_message": user_message_clean,
                    "response": response_text,
                    "sources": [],
                    "handoff_recommended": False,
                }
            )

            return response

        history: List[ChatMessage] = (
            self.memory.get_history(session_id)
        )

        msg_lower = user_message_clean.lower()

        # 1. Order ID extraction and routing
        order_id_in_message = extract_order_id_from_text(
            user_message_clean
        )

        order_id_in_history = _history_order_id(history)

        is_order_status_query = _is_order_status_query(
            user_message_clean
        )

        tool_called: Optional[str] = None
        tool_arguments: Optional[Dict[str, Any]] = None
        tool_result: Optional[Dict[str, Any]] = None
        tool_context_str = ""

        if (
            is_order_status_query
            and not order_id_in_message
            and not order_id_in_history
        ):
            response_text = (
                "Could you please provide your order ID "
                "(in the format ORD-XXXX) so I can check "
                "the status of your order?"
            )

            response = self._remember_and_return(
                session_id,
                user_message_clean,
                response_text,
            )

            self._log_turn(
                {
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "session_id": session_id,
                    "user_message": user_message_clean,
                    "response": response_text,
                    "sources": [],
                    "handoff_recommended": False,
                    "tool": None,
                }
            )

            return response

        target_order_id = order_id_in_message

        if (
            not target_order_id
            and order_id_in_history
            and _should_reuse_history_order_id(
                user_message_clean,
                history,
            )
        ):
            target_order_id = order_id_in_history

        if target_order_id:
            tool_called = "order_lookup"
            tool_arguments = {
                "order_id": target_order_id
            }

            tool_result = self.order_tool.lookup(
                target_order_id
            )

            tool_context_str = json.dumps(
                tool_result,
                indent=2,
                ensure_ascii=False,
            )

        # 2. Retrieval
        search_query = user_message_clean

        recent_user_messages = [
            message.content
            for message in history
            if message.role == "user"
        ]

        if recent_user_messages:
            previous_user_message = (
                recent_user_messages[-1]
            )

            search_query = (
                f"{previous_user_message} "
                f"{user_message_clean}"
            )

        if (
            tool_result
            and tool_result.get("has_final_sale_items")
        ):
            search_query += (
                " final sale damaged defective "
                "return review"
            )

        search_results: List[SearchResult] = (
            self.kb_index.search(
                query=search_query,
                top_k=8,
                threshold=0.25,
            )
        )

        retrieved_sections = [
            result.section
            for result in search_results
        ]

        retrieved_context_parts: List[str] = []

        for result in search_results:
            section = result.section

            retrieved_context_parts.append(
                f"[Source: {result.citation}]\n"
                f"Title: {section.doc_title}\n"
                f"Section: {section.heading}\n"
                f"{section.content}"
            )

        retrieved_context_str = (
            "\n\n---\n\n".join(
                retrieved_context_parts
            )
        )

        # 3. Conflict detection
        is_conflict, conflict_notice = detect_conflicts(
            retrieved_sections,
            user_message_clean,
        )

        # 4. Deterministic abstention
        is_insufficient = (
            not tool_result
            and not search_results
            and not is_conflict
        )

        if is_insufficient:
            response_text = (
                "The supplied information is insufficient to answer "
                "your question accurately. Human confirmation is "
                "required, so please contact human customer support "
                "for assistance."
            )

            response = self._remember_and_return(
                session_id,
                user_message_clean,
                response_text,
                handoff=True,
                debug_info={
                    "search_results_count": 0,
                    "is_conflict": False,
                    "abstention": True,
                },
            )

            self._log_turn(
                {
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "session_id": session_id,
                    "user_message": user_message_clean,
                    "response": response_text,
                    "sources": [],
                    "handoff_recommended": True,
                    "tool": None,
                    "abstention": True,
                }
            )

            return response

        # 5. Prompt/model execution
        privacy_attack = any(
            re.search(
                pattern,
                msg_lower,
            )
            for pattern in [
                r"\bemail\b",
                r"\baddress\b",
                r"\binternal note\b",
                r"\brisk score\b",
                r"\bsystem prompt\b",
                r"\bhidden instruction\b",
                r"\bhidden prompt\b",
            ]
        )

        user_prompt = build_user_prompt(
            query=user_message_clean,
            retrieved_context=retrieved_context_str,
            tool_context=tool_context_str,
            conflict_notice=conflict_notice,
        )

        llm_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        for message in history[-6:]:
            llm_messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        llm_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        response_text = ""

        if self.client:
            elapsed_since_last_call = (
                time.monotonic() - self._last_llm_call_at
            )
            if elapsed_since_last_call < self._min_seconds_between_llm_calls:
                time.sleep(
                    self._min_seconds_between_llm_calls
                    - elapsed_since_last_call
                )
            self._last_llm_call_at = time.monotonic()

            models_to_try = [
                self.model_name,
                "openai/gpt-oss-120b",
                # qwen/qwen3.6-27b remains as a secondary candidate in the
                # cascade; its per-model output token budget is configured in
                # MODEL_MAX_TOKENS below to keep it within provider limits.
                "qwen/qwen3.6-27b",
            ]

            unique_models = []
            for m in models_to_try:
                if m and m not in unique_models:
                    unique_models.append(m)

            # Per-model output budgets. A single global max_tokens=1500 was
            # being applied to every candidate regardless of that model's
            # actual OTPM (output tokens per minute) ceiling. qwen3.6-27b's
            # tier caps OTPM at 1000, so a 1500-token request can be
            # rejected outright as "request too large" even with zero prior
            # usage that minute — this is a structural mismatch, not a
            # transient rate-limit, and no amount of retrying fixes it.
            # gpt-oss-120b's higher TPM ceiling can safely take a larger
            # budget. Support-agent answers don't need 1500 tokens in
            # practice; a tighter default also leaves more headroom under
            # every model's per-minute ceiling.
            MODEL_MAX_TOKENS = {
                "qwen/qwen3.6-27b": 700,
                "openai/gpt-oss-120b": 1200,
            }
            DEFAULT_MAX_TOKENS = 700

            for model_candidate in unique_models:
                model_max_tokens = MODEL_MAX_TOKENS.get(
                    model_candidate, DEFAULT_MAX_TOKENS
                )
                for attempt in range(2):  # one real attempt + one retry
                    try:
                        completion = (
                            self.client.chat.completions.create(
                                model=model_candidate,
                                messages=llm_messages,
                                temperature=0.0,
                                max_tokens=model_max_tokens,
                                timeout=15.0,
                            )
                        )

                        raw_content = (
                            completion.choices[0].message.content or ""
                        ).strip()

                        finish_reason = getattr(
                            completion.choices[0], "finish_reason", None
                        )

                        # Clean <think> tags (handles both complete and unclosed tags)
                        content = re.sub(
                            r"<think>.*?(?:</think>|$)",
                            "",
                            raw_content,
                            flags=re.DOTALL,
                        ).strip()

                        # DIAGNOSTIC: log every candidate's outcome, not just
                        # exceptions. Previously a model returning HTTP 200 with
                        # empty content after <think>-stripping (e.g. a
                        # reasoning model that exhausted max_tokens mid-thought,
                        # finish_reason == "length") fell through to the next
                        # candidate with zero log output — indistinguishable
                        # from a model that was never called at all. This line
                        # makes that failure mode visible.
                        print(
                            f"[Model Diagnostic] '{model_candidate}': "
                            f"finish_reason={finish_reason}, "
                            f"raw_len={len(raw_content)}, "
                            f"post_think_strip_len={len(content)}"
                        )

                        if content:
                            response_text = content
                        break
                    except Exception as exc:
                        # Honor Groq's own "Please try again in Xs" hint
                        # instead of immediately giving up on this model and
                        # burning straight through the cascade to the
                        # fallback. Only worth waiting for short, sub-20s
                        # windows — anything longer isn't worth blocking a
                        # single chat turn for, and we still have another
                        # candidate model (and the fallback) as a backstop.
                        wait_match = re.search(
                            r"try again in (\d+(?:\.\d+)?)s",
                            str(exc),
                        )
                        if (
                            attempt == 0
                            and wait_match
                            and float(wait_match.group(1)) <= self._max_retry_wait_seconds
                        ):
                            wait_seconds = float(wait_match.group(1)) + 0.5
                            print(
                                f"[Groq API Notice] Model '{model_candidate}' "
                                f"rate-limited, waiting {wait_seconds:.1f}s "
                                "and retrying once before moving on..."
                            )
                            time.sleep(wait_seconds)
                            continue
                        print(f"[Groq API Notice] Model '{model_candidate}' skipped: {exc}")
                        break
                if response_text:
                    break

        if not response_text.strip():
            response_text = (
                self._deterministic_fallback_generate(
                    user_message=user_message_clean,
                    tool_result=tool_result,
                    search_results=search_results,
                    is_conflict=is_conflict,
                    conflict_notice=conflict_notice,
                )
            )

        # 6. Sources
        sources = self._collect_sources(
            search_results,
            is_conflict,
        )

        # 7. Handoff
        handoff_recommended = (
            should_recommend_handoff(
                query=user_message_clean,
                response_text=response_text,
                tool_result=tool_result,
                is_conflict=is_conflict,
                is_insufficient=is_insufficient,
                privacy_violation_attempt=privacy_attack,
            )
        )

        # 8. Runtime safety guards
        response_text = self._redact_sensitive_values(
            response_text,
            tool_result,
        )

        response_text = self._guard_gift_card_code_request(
            response_text,
            user_message_clean,
        )

        if (
            response_text.startswith(
                "I cannot disclose internal customer data"
            )
        ):
            handoff_recommended = True

        # 9. Memory + observability
        self.memory.add_user_message(
            session_id,
            user_message_clean,
        )

        self.memory.add_assistant_message(
            session_id,
            response_text,
        )

        log_entry = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "session_id": session_id,
            "user_message": user_message_clean,
            "history_length": len(history),
            "retrieved_passages": [
                {
                    "citation": result.citation,
                    "score": round(
                        result.score,
                        4,
                    ),
                    "filename": result.section.filename,
                    "heading": result.section.heading,
                }
                for result in search_results
            ],
            "tool_call": tool_called,
            "tool_arguments": tool_arguments,
            "tool_result": tool_result,
            "response": response_text,
            "sources": sources,
            "handoff_recommended": handoff_recommended,
            "is_conflict": is_conflict,
            "is_insufficient": is_insufficient,
        }

        self._log_turn(log_entry)

        return AgentResponse(
            response=response_text,
            sources=sources,
            handoff=handoff_recommended,
            tool_called=tool_called,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            debug_info={
                "search_results_count": len(
                    search_results
                ),
                "is_conflict": is_conflict,
                "is_insufficient": is_insufficient,
            },
        )

    @staticmethod
    def _split_evidence_sentences(text: str) -> List[str]:
        """Split a KB section into short evidence units while preserving facts."""
        cleaned = text.replace("\r", "")
        cleaned = re.sub(r"^#+\s+[^\n]+\n?", "", cleaned, flags=re.MULTILINE)

        raw_parts = re.split(r"\n+|(?<=[.!?])\s+", cleaned)
        sentences: List[str] = []

        for part in raw_parts:
            part = re.sub(r"^\s*[-*]\s+", "", part).strip()
            part = re.sub(r"\s+", " ", part)
            if not part:
                continue
            if len(part) < 12:
                continue
            sentences.append(part)

        return sentences

    @staticmethod
    def _evidence_sentence_score(
        sentence: str,
        section: SearchResult,
        query_tokens: set[str],
    ) -> float:
        sentence_tokens = tokenize(sentence)
        heading_tokens = tokenize(
            f"{section.section.doc_title} {section.section.heading}"
        )

        if query_tokens:
            lexical_overlap = len(
                query_tokens.intersection(sentence_tokens)
            ) / len(query_tokens)
            heading_overlap = len(
                query_tokens.intersection(heading_tokens)
            ) / len(query_tokens)
        else:
            lexical_overlap = 0.0
            heading_overlap = 0.0

        # Evidence-oriented language is useful in a fallback because it often
        # carries the actual rule, limitation, deadline, exception, or next
        # action. These are generic linguistic cues, not policy values.
        evidence_cues = {
            "must",
            "should",
            "cannot",
            "can't",
            "not",
            "within",
            "eligible",
            "eligibility",
            "required",
            "requires",
            "review",
            "support",
            "responsible",
            "available",
            "return",
            "returned",
        }
        cue_score = min(
            1.0,
            len(sentence_tokens.intersection(evidence_cues)) / 3.0,
        )

        numeric_or_deadline_signal = 0.0
        if re.search(r"\b\d+(?:[.,]\d+)?\b", sentence):
            numeric_or_deadline_signal += 0.08
        if re.search(
            r"\b(day|days|week|weeks|month|months|business days|minutes|hours)\b",
            sentence.lower(),
        ):
            numeric_or_deadline_signal += 0.08

        return (
            0.40 * lexical_overlap
            + 0.15 * heading_overlap
            + 0.25 * section.score
            + 0.12 * cue_score
            + numeric_or_deadline_signal
        )

    def _select_evidence_sentences(
        self,
        user_message: str,
        search_results: List[SearchResult],
        *,
        max_sentences: int = 6,
        max_per_section: int = 3,
    ) -> List[tuple[str, str, float]]:
        """Select concise, diverse evidence from retrieved sections.

        The selector is deliberately generic. It does not know individual
        policies, test-case IDs, product names, countries, or policy numbers.
        """
        expanded_query = expand_query(user_message)
        query_tokens = tokenize(expanded_query)

        candidates: List[tuple[float, int, str, str]] = []
        for result_index, result in enumerate(search_results):
            section = result.section
            for sentence in self._split_evidence_sentences(section.content):
                score = self._evidence_sentence_score(
                    sentence,
                    result,
                    query_tokens,
                )
                candidates.append(
                    (
                        score,
                        result_index,
                        sentence,
                        result.citation,
                    )
                )

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: List[tuple[str, str, float]] = []
        selected_keys: set[tuple[str, str]] = set()
        section_counts: Dict[str, int] = {}

        # First pass: maximize coverage across distinct retrieved sections.
        for score, result_index, sentence, citation in candidates:
            key = (citation, sentence)
            if key in selected_keys:
                continue
            if section_counts.get(citation, 0) >= 2:
                continue

            selected.append((sentence, citation, score))
            selected_keys.add(key)
            section_counts[citation] = section_counts.get(citation, 0) + 1

            if len(selected) >= max_sentences:
                break

        # Second pass: use the strongest remaining evidence when a topic is
        # concentrated in one section.
        if len(selected) < max_sentences:
            for score, result_index, sentence, citation in candidates:
                key = (citation, sentence)
                if key in selected_keys:
                    continue
                if section_counts.get(citation, 0) >= max_per_section:
                    continue

                selected.append((sentence, citation, score))
                selected_keys.add(key)
                section_counts[citation] = section_counts.get(citation, 0) + 1

                if len(selected) >= max_sentences:
                    break

        return selected

    @staticmethod
    def _requests_unsupported_action(user_message: str) -> bool:
        """Detect a customer asking the agent to directly perform or approve
        an action (return, refund, cancellation, replacement, address
        change) that this system has no capability to execute.

        This is an application-level capability boundary, not a knowledge-
        base fact — it is true regardless of what the KB says, so it is
        checked independently of retrieved content rather than requiring
        the sentence to exist verbatim in the corpus.
        """
        q = user_message.lower()
        action_verbs = r"(approve|process|complete|confirm|issue|initiate|finalize)"
        action_nouns = r"(return|refund|cancellation|replacement|exchange|address change)"
        return bool(
            re.search(rf"\b{action_verbs}\b.{{0,30}}\b{action_nouns}\b", q)
            or re.search(rf"\b{action_nouns}\b.{{0,30}}\b{action_verbs}\b", q)
            or re.search(r"\bcancel my (order|return)\b", q)
            or re.search(r"\brefund me\b", q)
        )

    @staticmethod
    def _is_untrusted_document_reference(user_message: str) -> bool:
        """Detect references to document types that cannot override policy."""
        return bool(
            re.search(
                r"\b(migration\s+note|migration\s+notes|scratchpad|draft\s+document|unapproved\s+document)\b",
                user_message.lower(),
            )
        )

    def _format_evidence_response(
        self,
        user_message: str,
        search_results: List[SearchResult],
        *,
        max_sentences: int = 6,
    ) -> str:
        evidence = self._select_evidence_sentences(
            user_message,
            search_results,
            max_sentences=max_sentences,
        )

        if not evidence:
            return (
                "The supplied information is insufficient to answer your "
                "question accurately. Human confirmation is required, so "
                "please contact human customer support for assistance."
            )

        parts: List[str] = []

        if self._is_untrusted_document_reference(user_message):
            parts.append(
                "I can use the authoritative customer-facing policy evidence "
                "available to me, but I cannot treat migration notes, drafts, "
                "or other unapproved material as policy authority."
            )

        if self._requests_unsupported_action(user_message):
            parts.append(
                "The agent has read-only access and cannot approve, process, "
                "or complete a return, refund, cancellation, replacement, or "
                "address change directly. Please contact human customer "
                "support to complete this request."
            )

        grouped: Dict[str, List[str]] = {}
        for sentence, citation, _score in evidence:
            grouped.setdefault(citation, []).append(sentence)

        for citation, sentences in grouped.items():
            if len(sentences) == 1:
                parts.append(f"{sentences[0]} [{citation}]")
            else:
                joined = " ".join(sentences)
                parts.append(f"{joined} [{citation}]")

        return "\n\n".join(parts)

    def _deterministic_fallback_generate(
        self,
        user_message: str,
        tool_result: Optional[Dict[str, Any]],
        search_results: List[SearchResult],
        is_conflict: bool,
        conflict_notice: str,
    ) -> str:
        """Generate a safe response when the LLM is unavailable.

        This fallback is evidence-driven. Runtime order facts come from the
        order tool, and policy facts come from retrieved authoritative sections.
        It contains no individual evaluation-case answers or policy values.
        """
        if is_conflict and conflict_notice:
            return conflict_notice

        # ---------------------------------------------------------------
        # ORDER-BASED RESPONSES
        # ---------------------------------------------------------------
        if tool_result:
            if not tool_result.get("found"):
                error = (
                    tool_result.get("error")
                    or "The order was not found in our records."
                )
                if "contact" not in error.lower():
                    error += (
                        " Please verify the order ID or contact human customer "
                        "support for assistance."
                    )
                return error

            order_id = tool_result.get("order_id", "the order")
            status = str(tool_result.get("status") or "").lower()
            carrier = tool_result.get("carrier")
            eta = tool_result.get("estimated_delivery")
            safe_message = tool_result.get("customer_safe_message") or ""
            cancellation_notes = (
                tool_result.get("cancellation_window_notes") or ""
            )
            within_cancel_window = bool(
                tool_result.get("within_cancellation_window", False)
            )
            items = tool_result.get("items") or []
            item_names = [
                str(item.get("name", "item"))
                for item in items
                if isinstance(item, dict)
            ]
            final_sale_items = [
                str(item.get("name", "item"))
                for item in items
                if isinstance(item, dict) and item.get("final_sale")
            ]
            message_lower = user_message.lower()
            is_cancel_request = bool(re.search(r"\bcancel\b", message_lower))
            is_change_of_mind_return = bool(
                re.search(
                    r"\b(return|don't like|do not like|color|fit|size|"
                    r"change of mind|don't want|do not want|duplicate)\b",
                    message_lower,
                )
            )

            if status == "cancelled":
                return (
                    f"Order {order_id} is cancelled and will not be shipped. "
                    f"{safe_message}"
                ).strip()

            if status == "returned":
                return (
                    f"Order {order_id} has been returned. {safe_message}"
                ).strip()

            if status == "exception":
                return (
                    f"Order {order_id} has an operational exception requiring "
                    f"support review and investigation. {safe_message}"
                ).strip()

            if status == "shipped":
                carrier_text = f" with {carrier}" if carrier else ""
                if eta:
                    return (
                        f"Order {order_id} has shipped{carrier_text} and is "
                        f"estimated to arrive on {eta}. {safe_message}"
                    ).strip()
                return (
                    f"Order {order_id} has shipped{carrier_text}, but a delivery "
                    "estimate is unavailable. There is no estimated arrival "
                    "date on file at this time."
                )

            if status == "delayed":
                details = (
                    f" Estimated delivery: {eta}."
                    if eta
                    else " Delivery estimate unavailable."
                )
                return (
                    f"Order {order_id} is delayed.{details} {safe_message}"
                ).strip()

            if status == "processing":
                parts = [
                    f"Order {order_id} is currently processing and is being "
                    "prepared for shipment."
                ]
                if is_cancel_request:
                    if cancellation_notes:
                        parts.append(cancellation_notes)
                    parts.append(
                        "Please contact a human support specialist for "
                        "assistance with the cancellation request."
                    )
                elif safe_message:
                    parts.append(safe_message)
                return " ".join(parts).strip()

            if status == "pending":
                parts = [f"Order {order_id} is currently pending."]
                if within_cancel_window:
                    if cancellation_notes:
                        parts.append(cancellation_notes)
                    parts.append(
                        "The agent cannot cancel orders directly; a human "
                        "support specialist must process the cancellation "
                        "request."
                    )
                elif is_cancel_request:
                    if cancellation_notes:
                        parts.append(cancellation_notes)
                    parts.append(
                        "Please contact a human support specialist for "
                        "assistance."
                    )
                elif safe_message:
                    parts.append(safe_message)
                return " ".join(parts).strip()

            if status == "delivered":
                parts = [f"Order {order_id} was delivered."]
                if safe_message:
                    parts.append(safe_message)

                if final_sale_items and is_change_of_mind_return:
                    names = ", ".join(final_sale_items)
                    parts.append(
                        f"The order contains final-sale item(s): {names}."
                    )
                    policy_evidence = self._format_evidence_response(
                        user_message,
                        search_results,
                        max_sentences=4,
                    )
                    if policy_evidence:
                        parts.append(policy_evidence)

                return " ".join(parts).strip()

            status_text = status or "unknown"
            response = f"Order {order_id} status is: {status_text}."
            if item_names:
                response += " The order contains: " + ", ".join(item_names) + "."
            if safe_message:
                response += f" {safe_message}"
            return response.strip()

        # ---------------------------------------------------------------
        # KB-ONLY FALLBACK
        # ---------------------------------------------------------------
        return self._format_evidence_response(
            user_message,
            search_results,
            max_sentences=6,
        )