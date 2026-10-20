import json
import logging
from typing import List, Dict, Any
from menu_api import MenuAPIClient
from cart import OrderCart

logger = logging.getLogger("Tools")

# Ollama Function Calling Schema
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_categories",
            "description": "Call when customer asks to see the menu, asks 'what is today to eat?', 'what's special today?', 'what do you have?', 'show me the orders/menu', or asks for food options. Returns active menu categories.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_products",
            "description": "Call when customer wants to view, see, browse, or explore products in a category or asks 'what items do you have in [category]', 'show me [category]', or 'I want to see all items in [category]' (e.g. 'see items in soft ice', 'show me cold drinks', 'what float options do you have'). NEVER call add_item when customer is asking to see or browse items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_id": {
                        "type": "integer",
                        "description": "The category ID number if known"
                    },
                    "category_name": {
                        "type": "string",
                        "description": "The category name (e.g. 'COLD DRINKS', 'HOT DRINKS', 'FLOAT', 'SOFT ICE')"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "Call ONLY when customer is inquiring about menu options or asking what items you have (e.g., 'what kind of drinks do you have?', 'do you have vegan items?'). NEVER call search_product when customer states what they want to order or eat—call add_item directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The category or item keyword to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add a specific item to the customer order cart when customer clearly requests to purchase, buy, or add a food or beverage (e.g. 'I will have a soft ice cone', 'add one latte', 'give me a cappuccino', 'two coffees', 'a croissant'). NEVER call add_item when customer says 'see', 'show', 'view', 'browse', or asks what items exist in a category—use get_products or search_product instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID"
                    },
                    "product_name": {
                        "type": "string",
                        "description": "Exact name of the product from menu search"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity to add (default 1)"
                    },
                    "price": {
                        "type": "number",
                        "description": "Price per unit as returned by menu tools"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Special instructions (e.g., 'no onions', 'extra ice')"
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove an item from the customer's active cart or reduce its quantity when customer asks to remove, cancel, or delete an item (e.g., 'remove the burger', 'take off the drink', 'cancel one fries', 'remove one item from cart').",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name or keyword of the item to remove (e.g. 'burger', 'latte', 'fries', 'last item')"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity to remove (default 1)"
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_cart",
            "description": "Clear or remove ALL items from the customer cart (e.g. 'remove everything from cart', 'clear my cart', 'empty the cart', 'start over', 'cancel all items'). Call this when customer wants to empty or clear their whole order tray.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "View items already placed inside the customer's active shopping cart (e.g. 'what did I order so far?', 'what is in my cart?'). Do NOT call this when customer is asking what food they can order or asking to see the menu.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_and_submit",
            "description": "Submit and place the finalized order ONLY after reading back items and total, and after customer gives a clear yes.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_item_notes",
            "description": "Update customization, sweetness level, ice level, or special instructions on an item in the cart (e.g. 'less sugar', 'no sugar', 'extra sugar', 'extra ice', 'no ice', 'oat milk'). Call this when the customer provides a modification or preference for an item without adding a new item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Special customization note (e.g. 'less sugar', 'no sugar', 'extra ice')"
                    },
                    "product_name": {
                        "type": "string",
                        "description": "Optional product name to customize. Defaults to the most recent item in cart."
                    }
                },
                "required": ["notes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_by_mood",
            "description": "Provide smart food and drink recommendations based on the customer's mood, state of mind, or occasion (e.g. 'tired/need energy', 'stressed/need to relax/comfort', 'hot day/refreshing', 'sweet/treat/happy', 'healthy/light').",
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "Customer mood or need (e.g. 'tired', 'energy', 'stressed', 'relax', 'refreshing', 'sweet', 'healthy')"
                    }
                },
                "required": ["mood"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_staff",
            "description": "Call human staff member for assistance when API fails, allergies/ingredients unknown, customer abusive or off-topic, or fails to understand twice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for calling staff (e.g. 'allergy question', 'API timeout', 'customer request')"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_addons",
            "description": "Call ONLY after add_item has been called for an add-on eligible item (Affogato, Americano Float, Americano Hot L/M, Cafe Latte Float, Cappuccino M/L, Caramel Banana Soft Ice, Caramel Latte Float, Caramel Latte Hot L/M). Returns the list of available add-ons (Non-Dairy Milk, Extra Fruit, Extra Sauce, Extra Soft, Extra Shot) each priced at 0.500 BHD.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_addon",
            "description": "Add an extra add-on (Non-Dairy Milk, Extra Fruit, Extra Sauce, Extra Soft, Extra Shot, each +0.500 BHD) to an eligible beverage or soft ice in the cart. Call this when customer agrees to an add-on or requests extra shot, non-dairy milk, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "addon_name": {
                        "type": "string",
                        "description": "Name of add-on: 'Non-Dairy Milk', 'Extra Fruit', 'Extra Sauce', 'Extra Soft', or 'Extra Shot'"
                    },
                    "product_name": {
                        "type": "string",
                        "description": "Optional name of the product in cart to attach add-on to"
                    }
                },
                "required": ["addon_name"]
            }
        }
    }
]


class ToolDispatcher:
    """Executes tool calls requested by the Chef LLM agent."""

    def __init__(self, menu_client: MenuAPIClient, cart: OrderCart):
        self.menu_client = menu_client
        self.cart = cart
        self.staff_call_history: List[Dict[str, Any]] = []

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool and return JSON string result."""
        logger.info(f"Dispatching tool: {tool_name} with args: {arguments}")

        try:
            if tool_name == "get_categories":
                res = self.menu_client.get_categories()
                return json.dumps(res)

            elif tool_name == "get_products":
                cat_id = arguments.get("category_id")
                cat_name = arguments.get("category_name")
                res = self.menu_client.get_products(category_id=cat_id, category_name=cat_name)
                return json.dumps(res)

            elif tool_name == "search_product":
                query = arguments.get("query", "")
                res = self.menu_client.search_product(query=query)
                return json.dumps(res)

            elif tool_name == "add_item":
                p_id = arguments.get("product_id", "")
                name = arguments.get("product_name", "")
                qty = int(arguments.get("quantity", 1))
                price = float(arguments.get("price", 0.0))
                notes = arguments.get("notes", "")

                # Auto-resolve product details from menu database if price or id is missing
                if price <= 0 or not p_id:
                    s_res = self.menu_client.search_product(name)
                    matches = s_res.get("products", [])
                    if matches:
                        top = matches[0]
                        p_id = p_id or top.get("productId")
                        name = top.get("name", name)
                        price = float(top.get("price", price))

                item = self.cart.add_item(
                    product_id=p_id,
                    name=name,
                    quantity=qty,
                    price=price,
                    notes=notes
                )

                result = {
                    "success": True,
                    "status": "added",
                    "item_added": item.name,
                    "quantity": item.quantity,
                    "unit_price": item.price,
                    "current_cart_total": self.cart.get_total(),
                    "items_in_cart": len(self.cart.items)
                }

                # If this item is add-on eligible, prompt the agent to ask about add-ons
                if self.menu_client.is_addon_eligible(item.name):
                    result["addon_eligible"] = True
                    result["addon_prompt"] = (
                        "This item supports add-ons. You MUST now call get_addons() to retrieve the list, "
                        "then ask the customer in ONE short sentence if they'd like any: "
                        "Non-Dairy Milk, Extra Fruit, Extra Sauce, Extra Soft, or Extra Shot (each +0.500 BHD). "
                        "Do NOT list all 5 in detail — just ask briefly if they want any add-ons."
                    )
                    # Auto-apply add-on if mentioned in notes (e.g. "with extra shot")
                    lower_notes = notes.lower()
                    for kw, a_name in [
                        ("non dairy milk", "Non-Dairy Milk"),
                        ("non-dairy milk", "Non-Dairy Milk"),
                        ("extra fruit", "Extra Fruit"),
                        ("extra sauce", "Extra Sauce"),
                        ("extra soft", "Extra Soft"),
                        ("extra shot", "Extra Shot"),
                    ]:
                        if kw in lower_notes:
                            self.cart.add_addon(addon_name=a_name, item_identifier=item.name, price=0.500)
                            result["addon_applied"] = a_name
                            result["current_cart_total"] = self.cart.get_total()
                            break

                return json.dumps(result)

            elif tool_name == "remove_item":
                name = arguments.get("product_name") or arguments.get("name") or arguments.get("item_name") or ""
                qty_val = arguments.get("quantity")
                qty = int(qty_val) if qty_val is not None else 1
                res = self.cart.remove_item(item_identifier=name, quantity=qty)
                return json.dumps(res)

            elif tool_name == "get_cart":
                return json.dumps({
                    "success": True,
                    "items": [
                        {"name": i.name, "quantity": i.quantity, "unit_price": i.price, "total": i.total_price, "notes": i.notes}
                        for i in self.cart.items
                    ],
                    "total": self.cart.get_total(),
                    "summary": self.cart.get_items_summary()
                })

            elif tool_name == "confirm_and_submit":
                if not getattr(self.cart, "confirmation_requested", False):
                    # Enforce Rule 5: read back items and total first
                    items_sum = self.cart.get_spoken_items_summary()
                    total = self.cart.get_total()
                    self.cart.confirmation_requested = True
                    return json.dumps({
                        "success": False,
                        "action_required": f"DO NOT submit yet. You must first read back the items ({items_sum}) and total ({total:.3f} BHD), then ask: 'Shall I place this order?'. Wait for customer's clear yes before calling confirm_and_submit."
                    })
                res = self.cart.confirm_and_submit()
                self.cart.confirmation_requested = False
                return json.dumps(res)

            elif tool_name == "update_item_notes":
                notes = arguments.get("notes", "")
                p_name = arguments.get("product_name")
                res = self.cart.update_item_notes(notes=notes, item_identifier=p_name)
                return json.dumps(res)

            elif tool_name == "recommend_by_mood":
                mood_input = arguments.get("mood", "")
                res = self._tool_recommend_by_mood(mood_input)
                return json.dumps(res)

            elif tool_name == "get_addons":
                p_name = arguments.get("product_name")
                res = self.menu_client.get_addons(item_name=p_name)
                return json.dumps(res)

            elif tool_name == "add_addon":
                addon_name = arguments.get("addon_name") or arguments.get("name") or ""
                p_name = arguments.get("product_name")
                # First check eligibility of target item
                target_item = None
                if p_name:
                    for it in self.cart.items:
                        if p_name.lower() in it.name.lower() or it.name.lower() in p_name.lower():
                            target_item = it
                            break
                elif self.cart.items:
                    target_item = self.cart.items[-1]

                if not target_item:
                    return json.dumps({
                        "success": False,
                        "error": "The cart is empty. Please add an item first."
                    })

                if not self.menu_client.is_addon_eligible(target_item.name):
                    return json.dumps({
                        "success": False,
                        "addon_eligible": False,
                        "message": f"Sorry, add-ons are only available for select items (such as Affogato, Floats, and Cappuccinos), not for '{target_item.name}'."
                    })

                res = self.cart.add_addon(addon_name=addon_name, item_identifier=target_item.name, price=0.500)
                return json.dumps(res)

            elif tool_name == "clear_cart":
                count = len(self.cart.items)
                self.cart.clear()
                return json.dumps({
                    "success": True,
                    "cleared_all": True,
                    "items_removed": count,
                    "message": "I have removed everything from your cart. Your order tray is now empty."
                })

            elif tool_name == "call_staff":
                reason = arguments.get("reason", "Customer requested assistance")
                self.staff_call_history.append({"reason": reason})
                return json.dumps({
                    "success": True,
                    "status": "staff_alerted",
                    "message": "Staff has been notified and is coming to assist at the kiosk."
                })

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return json.dumps({"error": f"Execution error in {tool_name}: {str(e)}"})

    def _tool_recommend_by_mood(self, mood: str) -> Dict[str, Any]:
        """Provide tailored recommendations matching the customer's mood, state of mind, or craving."""
        mood_lower = (mood or "").lower()
        if any(w in mood_lower for w in ["tired", "sleepy", "energy", "wake", "boost", "morning", "drowsy"]):
            return {
                "success": True,
                "mood_category": "energy_boost",
                "headline": "Energizing picks to power your day",
                "recommendations": [
                    {"name": "Espresso Single", "price": 2.50, "vibe": "Bold, intense espresso shot for an instant wake-up"},
                    {"name": "Cold Brew Coffee", "price": 4.25, "vibe": "Slow-steeped high-caffeine refreshing coffee"},
                    {"name": "Hot Cappuccino", "price": 4.00, "vibe": "Rich velvety espresso with frothy steamed milk"}
                ]
            }
        elif any(w in mood_lower for w in ["stress", "relax", "calm", "rough", "break", "unwind", "chill", "peace"]):
            return {
                "success": True,
                "mood_category": "relaxing_comfort",
                "headline": "Soothing comfort to help you unwind",
                "recommendations": [
                    {"name": "Hot Green Tea", "price": 3.00, "vibe": "Calming, fragrant authentic Japanese green tea"},
                    {"name": "Butter Croissant", "price": 3.25, "vibe": "Warm, flaky fresh-baked comfort pastry"},
                    {"name": "Chocolate Milkshake", "price": 5.25, "vibe": "Decadent creamy chocolate comfort treat"}
                ]
            }
        elif any(w in mood_lower for w in ["hot", "thirsty", "refresh", "summer", "cool", "heat"]):
            return {
                "success": True,
                "mood_category": "cool_refreshing",
                "headline": "Crisp and thirst-quenching options",
                "recommendations": [
                    {"name": "Fresh Apple Juice", "price": 3.75, "vibe": "Chilled, crisp 100% natural apple juice"},
                    {"name": "Iced Lemon Tea", "price": 3.00, "vibe": "Refreshing black tea infused with fresh lemon slices"},
                    {"name": "Cola Ice Cream Float", "price": 4.50, "vibe": "Chilled fizzy cola with creamy vanilla soft serve"}
                ]
            }
        elif any(w in mood_lower for w in ["sweet", "treat", "happy", "dessert", "celebrate", "craving", "indulge"]):
            return {
                "success": True,
                "mood_category": "sweet_indulgence",
                "headline": "Delicious sweet treats to delight you",
                "recommendations": [
                    {"name": "Strawberry Kakigori", "price": 6.00, "vibe": "Fluffy Japanese shaved ice with strawberry and condensed cream"},
                    {"name": "Twist Soft Ice Cup", "price": 3.50, "vibe": "Creamy swirl of vanilla and chocolate soft serve"},
                    {"name": "Classic Affogato", "price": 4.50, "vibe": "Hot espresso poured over rich vanilla ice cream"}
                ]
            }
        elif any(w in mood_lower for w in ["health", "diet", "light", "clean", "fresh", "guilt"]):
            return {
                "success": True,
                "mood_category": "healthy_light",
                "headline": "Fresh, light, and feel-good choices",
                "recommendations": [
                    {"name": "Hot Green Tea", "price": 3.00, "vibe": "Zero-calorie, antioxidant-rich pure green tea"},
                    {"name": "Fresh Apple Juice", "price": 3.75, "vibe": "Naturally sweet, vitamin-rich freshly pressed juice"},
                    {"name": "Sparkling Water", "price": 2.25, "vibe": "Zero-calorie crisp sparkling mineral water"}
                ]
            }
        else:
            return {
                "success": True,
                "mood_category": "customer_favorites",
                "headline": "Our top customer favorites",
                "recommendations": [
                    {"name": "Hot Latte", "price": 4.25, "vibe": "Our top-selling smooth steamed espresso latte"},
                    {"name": "Iced Caramel Latte", "price": 4.75, "vibe": "Chilled rich espresso with decadent caramel"},
                    {"name": "Fresh Apple Juice", "price": 3.75, "vibe": "Fresh and revitalizing fruit juice"}
                ]
            }
