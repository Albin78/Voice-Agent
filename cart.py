from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random
import re
import time


@dataclass
class CartItem:
    product_id: Any
    name: str
    quantity: int = 1
    price: float = 0.0
    notes: str = ""
    addons: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def addon_total(self) -> float:
        return sum(float(a.get("price", 0.500)) for a in self.addons)

    @property
    def unit_total(self) -> float:
        return round(self.price + self.addon_total, 3)

    @property
    def total_price(self) -> float:
        return round(self.quantity * self.unit_total, 3)


class OrderCart:
    """Manages active customer order session, items, prices, and suggestions."""

    def __init__(self):
        self.items: List[CartItem] = []
        self.last_quoted_total: Optional[float] = None
        self.declined_suggestions: set = set()
        self.is_submitted: bool = False
        self.order_number: Optional[str] = None
        self.confirmation_requested: bool = False  # Tracks if Chef has read back items before submit
        self.last_placed_order: Optional[Dict[str, Any]] = None

    def add_item(self, product_id: Any, name: str, quantity: int = 1, price: float = 0.0, notes: str = "") -> CartItem:
        self.confirmation_requested = False
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

    def set_item_quantity(self, quantity: int, item_identifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Set absolute quantity on a specific or most recent item (e.g. 'make it 2', 'change it to 3').
        If quantity is 0 or less, removes the item.
        """
        if not self.items:
            return {
                "success": False,
                "error": "The cart is empty. Please order an item first."
            }

        target_idx = -1
        if item_identifier:
            target = item_identifier.strip().lower()
            for i, item in enumerate(self.items):
                if target in item.name.lower() or item.name.lower() in target:
                    target_idx = i
                    break

        if target_idx == -1:
            target_idx = len(self.items) - 1

        item = self.items[target_idx]
        old_qty = item.quantity
        new_qty = int(quantity)

        if new_qty <= 0:
            removed_name = item.name
            self.items.pop(target_idx)
            return {
                "success": True,
                "action": "removed",
                "product_name": removed_name,
                "old_quantity": old_qty,
                "new_quantity": 0,
                "message": f"Removed {removed_name} from your cart.",
                "cart_summary": self.get_items_summary(),
                "cart_total": self.get_total()
            }

        item.quantity = new_qty
        return {
            "success": True,
            "action": "set",
            "product_name": item.name,
            "old_quantity": old_qty,
            "new_quantity": new_qty,
            "message": f"Updated {item.name} quantity to {new_qty}.",
            "cart_summary": self.get_items_summary(),
            "cart_total": self.get_total()
        }

    def adjust_item_quantity(self, delta: int = 1, item_identifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Increment or decrement quantity of a specific or most recent item (e.g. 'add one more', 'give me 2 more').
        """
        if not self.items:
            return {
                "success": False,
                "error": "The cart is empty. Please order an item first."
            }

        target_idx = -1
        if item_identifier:
            target = item_identifier.strip().lower()
            for i, item in enumerate(self.items):
                if target in item.name.lower() or item.name.lower() in target:
                    target_idx = i
                    break

        if target_idx == -1:
            target_idx = len(self.items) - 1

        item = self.items[target_idx]
        old_qty = item.quantity
        new_qty = old_qty + delta

        if new_qty <= 0:
            removed_name = item.name
            self.items.pop(target_idx)
            return {
                "success": True,
                "action": "removed",
                "product_name": removed_name,
                "old_quantity": old_qty,
                "new_quantity": 0,
                "message": f"Removed {removed_name} from your cart.",
                "cart_summary": self.get_items_summary(),
                "cart_total": self.get_total()
            }

        item.quantity = new_qty
        return {
            "success": True,
            "action": "adjusted",
            "delta": delta,
            "product_name": item.name,
            "old_quantity": old_qty,
            "new_quantity": new_qty,
            "message": f"Added {delta} more {item.name}. New quantity: {new_qty}.",
            "cart_summary": self.get_items_summary(),
            "cart_total": self.get_total()
        }

    def add_addon(self, addon_name: str, item_identifier: Optional[str] = None, price: float = 0.500) -> Dict[str, Any]:
        """Add a specific add-on (Non-Dairy Milk, Extra Fruit, Extra Sauce, Extra Soft, Extra Shot) to an item."""
        if not self.items:
            return {
                "success": False,
                "error": "The cart is empty. Please order an eligible beverage or soft ice first before adding add-ons."
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

        # Standardize add-on name
        name_lower = addon_name.strip().lower()
        clean_name = addon_name.strip()
        if "non" in name_lower and ("milk" in name_lower or "dairy" in name_lower):
            clean_name = "Non-Dairy Milk"
        elif "fruit" in name_lower:
            clean_name = "Extra Fruit"
        elif "sauce" in name_lower or "syrup" in name_lower:
            clean_name = "Extra Sauce"
        elif "soft" in name_lower:
            clean_name = "Extra Soft"
        elif "shot" in name_lower or "espresso" in name_lower:
            clean_name = "Extra Shot"

        existing = [a.get("name") for a in item.addons]
        if clean_name in existing:
            return {
                "success": True,
                "already_added": True,
                "addon_name": clean_name,
                "product_name": item.name,
                "message": f"{clean_name} is already added to {item.name}.",
                "item_total": item.total_price,
                "cart_total": self.get_total()
            }

        addon_obj = {"name": clean_name, "price": round(float(price), 3)}
        item.addons.append(addon_obj)
        if item.notes:
            if clean_name.lower() not in item.notes.lower():
                item.notes = f"{item.notes}, {clean_name}"
        else:
            item.notes = clean_name

        return {
            "success": True,
            "addon_name": clean_name,
            "price": addon_obj["price"],
            "product_name": item.name,
            "item_total": item.total_price,
            "cart_total": self.get_total(),
            "message": f"Added {clean_name} (+{addon_obj['price']:.3f} BHD) to your {item.name}. New total: {self.get_total():.3f} BHD."
        }

    def update_item_notes(self, notes: str, item_identifier: Optional[str] = None) -> Dict[str, Any]:
        """Update or set notes (customization, sweetness, ice) on a cart item."""
        if not self.items:
            return {
                "success": False,
                "error": "The cart is empty, so customization cannot be applied."
            }

        # Check if notes contains an add-on request (e.g. "extra shot", "non dairy milk", "extra fruit")
        lower_notes = notes.lower()
        addon_keywords = [
            ("non dairy milk", "Non-Dairy Milk"),
            ("non-dairy milk", "Non-Dairy Milk"),
            ("extra fruit", "Extra Fruit"),
            ("extra sauce", "Extra Sauce"),
            ("extra soft", "Extra Soft"),
            ("extra shot", "Extra Shot"),
        ]
        detected_addon = None
        for kw, a_name in addon_keywords:
            if kw in lower_notes:
                detected_addon = a_name
                break

        if detected_addon:
            return self.add_addon(addon_name=detected_addon, item_identifier=item_identifier, price=0.500)

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
        Supports fuzzy matching, quantity extraction, and complete removal keywords.
        If item_identifier is 'last', 'last item', 'one item', or empty and cart has items,
        operates on the most recently added item.
        """
        self.confirmation_requested = False
        if not self.items:
            return {
                "success": False,
                "error": "The cart is already empty. There are no items to remove."
            }

        target = (item_identifier or "").strip().lower()
        matched_idx = -1

        # Check if customer asked to remove everything / clear cart
        if target in {
            "all", "everything", "all items", "all the items", "everything from cart",
            "everything from my cart", "all items from cart", "cart", "the cart", "my cart",
            "entire cart", "whole cart"
        }:
            item_count = len(self.items)
            self.clear()
            return {
                "success": True,
                "cleared_all": True,
                "items_removed": item_count,
                "removed_item": "everything",
                "message": "Removed everything from your cart. Your order tray is now empty.",
                "cart_summary": "Cart is currently empty.",
                "new_total": 0.0
            }

        WORD_TO_NUM = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

        # Check if complete removal was requested ("completely", "all", "all of them", "entirely")
        remove_completely = False
        if re.search(r'\b(?:completely|all|all\s+of\s+them|all\s+of\s+it|entirely|whole)\b', target):
            remove_completely = True
            target = re.sub(r'\b(?:completely|all|all\s+of\s+them|all\s+of\s+it|entirely|whole)\b', '', target).strip()

        # Strip common leading words like "the ", "my ", "a ", "an "
        clean_target = re.sub(r'^(?:the|my|a|an)\s+', '', target).strip()

        # Extract embedded quantity from identifier (e.g. "5 21ice Original Chocolate Soft Ice" or "five lattes")
        q_m = re.match(r'^(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:of\s+)?(?:the\s+)?(.+)$', clean_target, re.IGNORECASE)
        if q_m:
            q_str = q_m.group(1).lower()
            extracted_qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM[q_str]
            clean_target = q_m.group(2).strip()
            if quantity == 1 or quantity is None:
                quantity = extracted_qty

        target = clean_target.strip()

        # Check for generic "last item", "one item", or empty identifier
        if target in {"last", "last item", "one item", "item", "the item", "it", ""} and len(self.items) > 0:
            matched_idx = len(self.items) - 1
        else:
            target_words = set(w.rstrip('s') for w in re.findall(r'\w+', target) if len(w) > 1)
            scored_candidates = []

            for i, item in enumerate(self.items):
                i_name = item.name.lower()
                i_id = str(item.product_id).lower()
                i_words = set(w.rstrip('s') for w in re.findall(r'\w+', i_name) if len(w) > 1)

                score = 0
                # 1. Exact match on full name or ID
                if target == i_name or target == i_id:
                    score = 1000
                # 2. Substring match
                elif target in i_name:
                    score = 500 + 100 * (len(target) / len(i_name))
                elif i_name in target:
                    score = 400 + 100 * (len(i_name) / len(target))
                else:
                    # 3. Stemmed token overlap with Jaccard weighting
                    overlap = target_words & i_words
                    if overlap:
                        jaccard = len(overlap) / len(target_words | i_words)
                        score = 100 * jaccard + 50 * len(overlap)

                if score > 0:
                    scored_candidates.append((score, i, item))

            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                matched_idx = scored_candidates[0][1]

        if matched_idx == -1:
            return {
                "success": False,
                "error": f"Item '{item_identifier}' not found in cart. Current items: {self.get_items_summary()}"
            }

        item = self.items[matched_idx]

        if remove_completely or quantity is None or int(quantity) >= item.quantity:
            qty_removed = item.quantity
            self.items.pop(matched_idx)
            msg = f"Removed {item.name} completely from cart."
            removed_all = True
        else:
            qty_to_remove = int(quantity) if int(quantity) > 0 else 1
            item.quantity -= qty_to_remove
            qty_removed = qty_to_remove
            msg = f"Reduced {item.name} quantity by {qty_to_remove}. Remaining: {item.quantity}."
            removed_all = False

        return {
            "success": True,
            "removed_item": item.name,
            "quantity_removed": qty_removed,
            "removed_completely": removed_all,
            "message": msg,
            "cart_summary": self.get_items_summary(),
            "new_total": self.get_total()
        }

    def get_total(self) -> float:
        return round(sum(item.total_price for item in self.items), 3)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def get_items_summary(self) -> str:
        """Returns readable summary of items in cart."""
        if not self.items:
            return "Cart is currently empty."
        parts = []
        for item in self.items:
            addon_str = ""
            if getattr(item, "addons", None):
                addon_names = ", ".join(a.get("name") for a in item.addons)
                addon_str = f" with {addon_names}"
            parts.append(f"{item.quantity}x {item.name}{addon_str} at {item.unit_total:.3f} BHD each")
        return ", ".join(parts) + f" (Total: {self.get_total():.3f} BHD)"

    def get_spoken_items_summary(self) -> str:
        """Returns warm, natural spoken summary of items in cart for readback."""
        if not self.items:
            return "no items"
        parts = []
        for item in self.items:
            addon_str = ""
            if getattr(item, "addons", None):
                addon_names = ", ".join(a.get("name") for a in item.addons)
                addon_str = f" with {addon_names}"
            parts.append(f"{item.quantity} {item.name}{addon_str}")
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def record_declined_suggestion(self, suggestion_name: str):
        self.declined_suggestions.add(suggestion_name.strip().lower())

    def has_declined_suggestion(self, suggestion_name: str) -> bool:
        return suggestion_name.strip().lower() in self.declined_suggestions

    def confirm_and_submit(self) -> Dict[str, Any]:
        """Finalize order, generate order number and save placed order before resetting active cart."""
        if self.is_empty():
            return {
                "success": False,
                "error": "Cannot submit empty order. Please add at least one item first."
            }

        order_num = f"GIR-{random.randint(101, 999)}"
        total = self.get_total()
        items_ordered = [
            {
                "product_id": item.product_id,
                "name": item.name,
                "quantity": item.quantity,
                "price": item.price,
                "unit_total": item.unit_total,
                "total_price": item.total_price,
                "notes": item.notes,
                "addons": list(item.addons) if getattr(item, "addons", None) else []
            }
            for item in self.items
        ]

        self.order_number = order_num
        self.is_submitted = True
        self.confirmation_requested = False
        self.last_placed_order = {
            "order_number": order_num,
            "total": total,
            "items": items_ordered,
            "time": time.strftime("%H:%M:%S")
        }

        # Reset active cart tray now that order is secured
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cart to dictionary for frontend storage."""
        return {
            "items": [
                {
                    "product_id": str(i.product_id),
                    "name": i.name,
                    "quantity": i.quantity,
                    "price": i.price,
                    "total_price": i.total_price,
                    "notes": i.notes
                }
                for i in self.items
            ],
            "total_items": sum(i.quantity for i in self.items),
            "total_amount": round(self.get_total(), 2),
            "is_empty": self.is_empty(),
            "is_submitted": self.is_submitted,
            "order_number": self.order_number
        }

    def load_from_dict(self, data: Optional[Dict[str, Any]]):
        """Populate in-memory cart directly from frontend state without database query."""
        if not data or not isinstance(data, dict):
            return
        self.items = []
        raw_items = data.get("items", [])
        for item in raw_items:
            try:
                self.items.append(CartItem(
                    product_id=item.get("product_id"),
                    name=item.get("name", ""),
                    quantity=int(item.get("quantity", 1)),
                    price=float(item.get("price", 0.0)),
                    notes=item.get("notes", "")
                ))
            except Exception:
                pass
        self.is_submitted = bool(data.get("is_submitted", False))
        self.order_number = data.get("order_number")

