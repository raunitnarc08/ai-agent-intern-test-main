import pytest
from tools.order_lookup import get_order_lookup_tool, normalize_order_id, is_valid_order_id_format


@pytest.fixture
def order_tool():
    return get_order_lookup_tool()


def test_id_normalization():
    assert normalize_order_id("  ord-1007. ") == "ORD-1007"
    assert normalize_order_id("ORD-1007!") == "ORD-1007"
    assert normalize_order_id("ord-1007,") == "ORD-1007"


def test_near_miss_rejection(order_tool):
    res1 = order_tool.lookup("ORD-107")
    assert res1["found"] is False
    assert "Invalid order ID format" in res1["error"]

    res2 = order_tool.lookup("ORD-10007")
    assert res2["found"] is False

    res3 = order_tool.lookup("ord 1007")
    assert res3["found"] is False


def test_field_sanitization_and_injection_stripping(order_tool):
    # ORD-1005 contains warehouse_note injection
    res = order_tool.lookup("ORD-1005")
    assert res["found"] is True
    assert "warehouse_note" not in res
    assert "risk_score" not in res
    assert "customer" not in res
    assert "support_tags" not in res
    assert res["status"] == "delayed"
    assert "weather delay" in res["customer_safe_message"]


def test_cancellation_window_math(order_tool):
    # ORD-1001 placed 11:45:00Z on 2026-08-15, pending -> 15m <= 30m -> True
    res1001 = order_tool.lookup("ORD-1001")
    assert res1001["found"] is True
    assert res1001["within_cancellation_window"] is True

    # ORD-1002 placed 09:20:00Z on 2026-08-15, processing -> False
    res1002 = order_tool.lookup("ORD-1002")
    assert res1002["found"] is True
    assert res1002["within_cancellation_window"] is False


def test_stale_field_suppression(order_tool):
    # ORD-1004 cancelled -> ETA suppressed
    res1004 = order_tool.lookup("ORD-1004")
    assert res1004["found"] is True
    assert res1004["status"] == "cancelled"
    assert res1004["estimated_delivery"] is None
    assert res1004["carrier"] is None
