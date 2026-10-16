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
            "description": "Call when customer asks what categories of items are available or asks 'what do you have?'. Returns active categories.",
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
            "description": "Get available products for a specific category name or ID.",
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
            "description": "Search the menu for a product by name, keyword, or customer request (e.g., 'burger', 'coffee', 'latte', 'ice cream').",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The food or drink item name or keyword to search for"
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
            "description": "Add an item to the customer order cart once their choice and quantity are clear.",
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
            "name": "get_cart",
            "description": "View all items currently in the customer order and the total amount.",
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
                return json.dumps({
                    "success": True,
                    "status": "added",
                    "item_added": item.name,
                    "quantity": item.quantity,
                    "unit_price": item.price,
                    "current_cart_total": self.cart.get_total(),
                    "items_in_cart": len(self.cart.items)
                })

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
                    items_sum = self.cart.get_items_summary()
                    total = self.cart.get_total()
                    self.cart.confirmation_requested = True
                    return json.dumps({
                        "success": False,
                        "action_required": f"DO NOT submit yet. You must first read back the items ({items_sum}) and total (${total:.2f}), then ask: 'Shall I place this order?'. Wait for customer's clear yes before calling confirm_and_submit."
                    })
                res = self.cart.confirm_and_submit()
                self.cart.confirmation_requested = False
                return json.dumps(res)

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
