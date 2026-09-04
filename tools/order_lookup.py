from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from config import ORDERS_PATH, SNAPSHOT_DATETIME


def normalize_order_id(raw_id: str) -> str:
    """Normalize whitespace, uppercase, and strip trailing punctuation."""
    if not raw_id:
        return ""
    # Strip whitespace and common trailing punctuation
    cleaned = raw_id.strip().rstrip(".,!?:;\"'")
    return cleaned.upper()


def is_valid_order_id_format(order_id: str) -> bool:
    """Strictly validate format ORD-XXXX where XXXX is 4 digits."""
    return bool(re.match(r"^ORD-\d{4}$", order_id))


class OrderLookupTool:
    def __init__(self, orders_path: Path = ORDERS_PATH):
        self.orders_path = orders_path
        self.orders_by_id: Dict[str, Dict[str, Any]] = {}
        self.snapshot_at = SNAPSHOT_DATETIME
        self._load_orders()

    def _load_orders(self):
        if not self.orders_path.exists():
            return
        with open(self.orders_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        snapshot_str = data.get("snapshot_at")
        if snapshot_str:
            try:
                self.snapshot_at = datetime.fromisoformat(snapshot_str.replace("Z", "+00:00"))
            except Exception:
                self.snapshot_at = SNAPSHOT_DATETIME

        for order in data.get("orders", []):
            oid = order.get("order_id")
            if oid:
                self.orders_by_id[oid.upper()] = order

    def lookup(self, raw_order_id: str) -> Dict[str, Any]:
        """
        Perform a safe lookup of an order by ID.
        
        Guarantees:
        1. Never exposes sensitive PII (customer name, email, shipping address).
        2. Never exposes internal fields (risk_score, warehouse_note, support_tags).
        3. Suppresses stale delivery fields on cancelled/returned orders.
        4. Computes deterministic 30-minute cancellation window against snapshot_at.
        5. Strictly rejects malformed or near-miss order IDs without guessing.
        """
        normalized_id = normalize_order_id(raw_order_id)
        
        if not normalized_id:
            return {
                "found": False,
                "error": "Missing order ID. Please provide an order ID in the format ORD-XXXX (e.g. ORD-1007)."
            }

        if not is_valid_order_id_format(normalized_id):
            return {
                "found": False,
                "error": f"Invalid order ID format: '{raw_order_id}'. Expected format is ORD-XXXX (e.g. ORD-1007). We do not guess or fuzzy-match order numbers."
            }

        raw_order = self.orders_by_id.get(normalized_id)
        if not raw_order:
            return {
                "found": False,
                "order_id": normalized_id,
                "error": f"Order {normalized_id} was not found in our records. Please verify the order ID or contact human customer support."
            }

        status = str(raw_order.get("status", "unknown")).lower()
        placed_at_str = raw_order.get("placed_at")
        
        # 30-minute cancellation window math against snapshot_at
        within_cancellation_window = False
        cancellation_window_explanation = ""
        
        if placed_at_str:
            try:
                placed_dt = datetime.fromisoformat(placed_at_str.replace("Z", "+00:00"))
                elapsed_seconds = (self.snapshot_at - placed_dt).total_seconds()
                elapsed_minutes = elapsed_seconds / 60.0
                
                if status == "pending" and 0 <= elapsed_minutes <= 30.0:
                    within_cancellation_window = True
                    cancellation_window_explanation = (
                        f"Order was placed {int(elapsed_minutes)} minutes ago and is currently pending (within the 30-minute window). "
                        "Cancellation may still be requested, but requires a human specialist to process."
                    )
                elif status != "pending":
                    within_cancellation_window = False
                    cancellation_window_explanation = f"Order status is '{status}'. Cancellation is only possible while order is pending."
                else:
                    within_cancellation_window = False
                    cancellation_window_explanation = f"Order was placed {int(elapsed_minutes)} minutes ago, which exceeds the 30-minute cancellation window."
            except Exception:
                pass

        # Handle customer-safe items
        safe_items = []
        has_final_sale_items = False
        for item in raw_order.get("items", []):
            is_fs = bool(item.get("final_sale", False))
            if is_fs:
                has_final_sale_items = True
            safe_items.append({
                "name": item.get("name"),
                "quantity": item.get("quantity"),
                "final_sale": is_fs
            })

        # Handle stale tracking / estimated delivery on cancelled or returned orders
        is_terminal_inactive = status in ["cancelled", "returned"]
        carrier = None if is_terminal_inactive else raw_order.get("carrier")
        tracking_number = None if is_terminal_inactive else raw_order.get("tracking_number")
        estimated_delivery = None if is_terminal_inactive else raw_order.get("estimated_delivery")
        
        requires_handoff = (status == "exception")

        # Construct strictly sanitized output
        sanitized_result: Dict[str, Any] = {
            "found": True,
            "order_id": normalized_id,
            "membership_tier": raw_order.get("membership_tier", "standard"),
            "status": status,
            "status_updated_at": raw_order.get("status_updated_at"),
            "placed_at": placed_at_str,
            "shipped_at": raw_order.get("shipped_at"),
            "delivered_at": raw_order.get("delivered_at"),
            "carrier": carrier,
            "tracking_number": tracking_number,
            "estimated_delivery": estimated_delivery,
            "customer_safe_message": raw_order.get("customer_safe_message"),
            "items": safe_items,
            "has_final_sale_items": has_final_sale_items,
            "within_cancellation_window": within_cancellation_window,
            "cancellation_window_notes": cancellation_window_explanation,
            "requires_human_handoff": requires_handoff
        }

        return sanitized_result


# Global singleton instance
_global_order_tool: Optional[OrderLookupTool] = None


def get_order_lookup_tool() -> OrderLookupTool:
    global _global_order_tool
    if _global_order_tool is None:
        _global_order_tool = OrderLookupTool()
    return _global_order_tool
