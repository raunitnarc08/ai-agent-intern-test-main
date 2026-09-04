import pytest
from agent.orchestrator import AsterRowAgent


@pytest.fixture
def agent():
    return AsterRowAgent()


def test_conflict_detection_and_handoff(agent):
    res = agent.chat("test-sess-1", "Can I put the entire Breeze Tumbler in the dishwasher?")
    assert res.handoff is True
    assert "11-product-care.md" in res.sources
    assert "12-breeze-tumbler-product-card.md" in res.sources
    assert "conflict" in res.response.lower()


def test_deterministic_abstention(agent):
    res = agent.chat("test-sess-2", "Are all fabrics and adhesives in your bags vegan?")
    assert res.handoff is True
    assert "insufficient" in res.response.lower()
    assert len(res.sources) == 0


def test_missing_order_id_prompt(agent):
    res = agent.chat("test-sess-3", "Where is my order?")
    assert res.tool_called is None
    assert "order ID" in res.response
