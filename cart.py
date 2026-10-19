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
        self.confirmation_requested: bool = False  # Tracks if Chef has read back items before submit

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

    def update_item_notes(self, notes: str, item_identifier: Optional[str] = None) -> Dict[str, Any]:
        """Update or set notes (customization, sweetness, ice) on a cart item."""
        if not self.items:
            return {
                "success": False,
                "error": "The cart is empty, so customization cannot be applied."
            }

        target_idx = -1
        if item_identifier:
            target = item_identifier.strip().lower()
            for i, item in enumerate(self.items):
                if target in item.name.lower() or item.name.lower() in target:
                    target_idx = i
                    break

        # Default to the most recently added item
        if target_idx == -1:
            target_idx = len(self.items) - 1

        item = self.items[target_idx]
        item.notes = notes.strip()
        return {
            "success": True,
            "product_name": item.name,
            "notes": item.notes,
            "message": f"Updated {item.name} notes to: {item.notes}",
            "cart_summary": self.get_items_summary()
        }

    def remove_item(self, item_identifier: str = "", quantity: Optional[int] = 1) -> Dict[str, Any]:
        """
        Remove item by name, keyword, or id, or reduce its quantity.
        Supports fuzzy matching (e.g., 'burger' matches 'Classic Cheeseburger').
        If item_identifier is 'last', 'last item', 'one item', or empty and cart has items,
        operates on the most recently added item.
        """
        if not self.items:
            return {
                "success": False,
                "error": "The cart is already empty. There are no items to remove."
            }

        target = (item_identifier or "").strip().lower()
        matched_idx = -1

        # Check for generic "last item", "one item", or empty identifier
        if target in {"last", "last item", "one item", "item", "the item", "it", ""} and len(self.items) > 0:
            matched_idx = len(self.items) - 1
        else:
            # 1. Exact match on ID or full name
            for i, item in enumerate(self.items):
                if str(item.product_id).lower() == target or item.name.lower() == target:
                    matched_idx = i
                    break

            # 2. Substring match (e.g. 'burger' in 'Truffle Burger')
            if matched_idx == -1:
                for i, item in enumerate(self.items):
                    if target in item.name.lower() or item.name.lower() in target:
                        matched_idx = i
                        break

            # 3. Word-level token match (e.g. 'caramel' matches 'Iced Caramel Latte')
            if matched_idx == -1:
                target_words = set(target.split())
                for i, item in enumerate(self.items):
                    item_words = set(item.name.lower().split())
                    if target_words & item_words:
                        matched_idx = i
                        break

        if matched_idx == -1:
            return {
                "success": False,
                "error": f"Item '{item_identifier}' not found in cart. Current items: {self.get_items_summary()}"
            }

        item = self.items[matched_idx]
        qty_to_remove = int(quantity) if quantity is not None and int(quantity) > 0 else 1

        if item.quantity > qty_to_remove:
            item.quantity -= qty_to_remove
            msg = f"Reduced {item.name} quantity by {qty_to_remove}. Remaining: {item.quantity}."
            removed_all = False
        else:
            actual_removed = item.quantity
            self.items.pop(matched_idx)
            msg = f"Removed {item.name} completely from cart."
            removed_all = True

        return {
            "success": True,
            "removed_item": item.name,
            "quantity_removed": qty_to_remove,
            "removed_completely": removed_all,
            "message": msg,
            "cart_summary": self.get_items_summary(),
            "new_total": self.get_total()
        }

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
        self.confirmation_requested = False
