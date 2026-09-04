import re

def clean_think_tags(content: str) -> str:
    """Helper mirroring orchestrator.py cleaning logic."""
    return re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip()

def test_closed_think_tag():
    raw_output = "<think>Analyzing user query...</think>Our standard return window is 30 days."
    cleaned = clean_think_tags(raw_output)
    assert cleaned == "Our standard return window is 30 days."
    assert "<think>" not in cleaned

def test_unclosed_think_tag():
    raw_output = "<think>Truncated internal reasoning process without closing tag..."
    cleaned = clean_think_tags(raw_output)
    assert cleaned == ""
    assert "<think>" not in cleaned

def test_multiline_closed_think_tag():
    raw_output = "<think>\nStep 1: Check database.\nStep 2: Synthesize.\n</think>\nYour order has shipped."
    cleaned = clean_think_tags(raw_output)
    assert cleaned == "Your order has shipped."
    assert "<think>" not in cleaned