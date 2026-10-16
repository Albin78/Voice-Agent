from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random


@dataclass
class CartItem:
    product_id: Any
    name: str
    quantity: int = 1
    price: float = 0.0
    notes: str = ""

    @property
    def total_price(self) -> float:
        return round(self.quantity * self.price, 2)


class OrderCart:
    """Manages active customer order session, items, prices, and suggestions."""

    def __init__(self):
        self.items: List[CartItem] = []
        self.last_quoted_total: Optional[float] = None
        self.declined_suggestions: set = set()
        self.is_submitted: bool = False
        self.order_number: Optional[str] = None

    def add_item(self, product_id: Any, name: str, quantity: int = 1, price: float = 0.0, notes: str = "") -> CartItem:
        # Check if item with same ID/name already in cart
        for item in self.items:
            if str(item.product_id) == str(product_id) or item.name.lower() == name.lower():
                item.quantity += quantity
                if price > 0:
                    item.price = price
                if notes:
                    item.notes = (item.notes + " " + notes).strip()
                return item

        new_item = CartItem(
            product_id=product_id,
            name=name,
            quantity=max(1, quantity),
            price=float(price),
            notes=notes
        )
        self.items.append(new_item)
        return new_item

    def remove_item(self, item_identifier: str) -> bool:
        """Remove item by name or id."""
        target_name = item_identifier.lower()
        for i, item in enumerate(self.items):
            if str(item.product_id) == item_identifier or item.name.lower() == target_name:
                self.items.pop(i)
                return True
        return False

    def get_total(self) -> float:
        return round(sum(item.total_price for item in self.items), 2)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def get_items_summary(self) -> str:
        """Returns readable summary of items in cart."""
        if not self.items:
            return "Cart is currently empty."
        parts = []
        for item in self.items:
            note_str = f" ({item.notes})" if item.notes else ""
            parts.append(f"{item.quantity}x {item.name}{note_str} at ${item.price:.2f} each")
        return ", ".join(parts) + f" (Total: ${self.get_total():.2f})"

    def record_declined_suggestion(self, suggestion_name: str):
        self.declined_suggestions.add(suggestion_name.strip().lower())

    def has_declined_suggestion(self, suggestion_name: str) -> bool:
        return suggestion_name.strip().lower() in self.declined_suggestions

    def confirm_and_submit(self) -> Dict[str, Any]:
        """Finalize order, generate order number and reset cart."""
        if self.is_empty():
            return {
                "success": False,
                "error": "Cannot submit empty order. Please add at least one item first."
            }

        order_num = str(random.randint(101, 999))
        total = self.get_total()
        items_ordered = [
            {"product_id": item.product_id, "name": item.name, "quantity": item.quantity, "price": item.price}
            for item in self.items
        ]

        self.order_number = order_num
        self.is_submitted = True
        
        # Reset cart after generating result
        self.items = []
        self.last_quoted_total = None
        self.declined_suggestions.clear()

        return {
            "success": True,
            "order_number": order_num,
            "total": total,
            "items": items_ordered,
            "message": f"Order #{order_num} confirmed successfully."
        }

    def clear(self):
        self.items = []
        self.last_quoted_total = None
        self.declined_suggestions.clear()
        self.is_submitted = False
        self.order_number = None
