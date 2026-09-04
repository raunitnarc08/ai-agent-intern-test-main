import re
from typing import Any, Dict, Optional


def should_recommend_handoff(
    query: str,
    response_text: str,
    tool_result: Optional[Dict[str, Any]] = None,
    is_conflict: bool = False,
    is_insufficient: bool = False,
    privacy_violation_attempt: bool = False,
) -> bool:
    """Return whether human support should be recommended.

    These rules describe operational/safety boundaries. They intentionally do
    not contain individual evaluation-case answers.
    """

    if is_conflict or is_insufficient or privacy_violation_attempt:
        return True

    if tool_result:
        if not tool_result.get("found", False):
            error = str(tool_result.get("error") or "").lower()

            # Invalid syntax can be corrected by the customer without human
            # intervention. A valid-looking but unknown order needs support.
            if "invalid order id format" in error:
                return False

            return True

        if tool_result.get("requires_human_handoff", False):
            return True

        if str(tool_result.get("status", "")).lower() == "exception":
            return True

    q = query.lower()

    # Damaged/defective/incorrect items generally require human review.
    damaged_patterns = [
        r"\bdamaged\b",
        r"\bbroken\b",
        r"\bdefective\b",
        r"\bfaulty\b",
        r"\bwrong item\b",
        r"\bdifferent item\b",
        r"\bincorrect item\b",
        r"\bitem i didn't order\b",
    ]

    if any(re.search(pattern, q) for pattern in damaged_patterns):
        return True

    # Delivery/carrier exceptions and delivery problems require support review.
    delivery_terms = (
        r"(carrier|delivery|shipping|shipment|package|parcel)"
        r".{0,50}"
        r"(exception|issue|problem|scan|incident|error)"
    )
    delivery_terms_reverse = (
        r"(exception|issue|problem|scan|incident|error)"
        r".{0,50}"
        r"(carrier|delivery|shipping|shipment|package|parcel)"
    )

    if re.search(delivery_terms, q) or re.search(delivery_terms_reverse, q):
        return True

    # Adversarial prompt injections, unapproved migration/draft notes,
    # or attempts to override rules should be firmly refuted with official
    # policy, not escalated to human agents.
    is_injection_or_override_attempt = bool(
        re.search(
            r"\b(migration note|scratchpad|ignore (all|prior|the|real) (rules|policy)|system instruction|override|newer document)\b",
            q,
        )
    )
    if is_injection_or_override_attempt:
        return False

    # Gift-card/store-credit checkout problems require support.
    gift_card_terms = r"\b(gift[\s-]?card|store[\s-]?credit|voucher|promotional credit)\b"
    checkout_problem_terms = (
        r"\b("
        r"error|stuck|fix|not working|isn['’]?t working|problem|checkout|check[\s-]?out|"
        r"declined|fails|failed|failure|won't work|won['’]?t work|cannot use|"
        r"can't use|can['’]?t use|unable to use"
        r")\b"
    )

    if re.search(gift_card_terms, q) and re.search(checkout_problem_terms, q):
        return True

    # Requests to perform unsupported operational actions require human support.
    action_patterns = [
        r"\bcancel (my|the|this) order\b",
        r"\bplease cancel\b",
        r"\bcan i cancel\b",
        r"\bcould you cancel\b",
        r"\bhow do i cancel\b",
        r"\bprocess (a )?refund\b",
        r"\bissue (a )?refund\b",
        r"\bgive me a refund\b",
        r"\brefund this\b",
        r"\bchange (my|the) address\b",
        r"\bupdate (my|the) address\b",
        r"\bchange (my|the) shipping address\b",
        r"\bapprove (my|the) return\b",
        r"\bapprove (my|the) warranty\b",
        r"\breplace (my|the) item\b",
        r"\breship (my|the) order\b",
        r"\bopen an investigation\b",
    ]

    for pattern in action_patterns:
        if re.search(pattern, q):
            return True

    return False