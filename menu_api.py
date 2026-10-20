import requests
import json
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger("MenuAPI")

# Real menu items matching the live API products (fallback when API token expires)
SAMPLE_PRODUCTS = [
    # COLD DRINKS (id: 19)
    {"productId": 101, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Iced Americano", "price": 1.500, "description": "Bold espresso over ice and cold water", "isActive": True},
    {"productId": 102, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Iced Caramel Latte", "price": 1.750, "description": "Rich espresso with chilled milk and caramel syrup", "isActive": True},
    {"productId": 103, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Cold Brew Coffee", "price": 1.750, "description": "Slow-steeped smooth iced coffee", "isActive": True},
    {"productId": 104, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Iced Lemon Tea", "price": 1.250, "description": "Refreshing black tea with fresh lemon slices", "isActive": True},
    {"productId": 122, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Fresh Apple Juice", "price": 1.500, "description": "Freshly pressed chilled apple juice", "isActive": True},

    # HOT DRINKS (id: 18)
    {"productId": 105, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Americano Hot M", "price": 1.250, "description": "Medium hot espresso americano", "isActive": True},
    {"productId": 106, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Americano Hot L", "price": 1.500, "description": "Large hot espresso americano", "isActive": True},
    {"productId": 107, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Cappuccino M", "price": 1.500, "description": "Medium cappuccino with steamed milk foam", "isActive": True},
    {"productId": 108, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Cappuccino L", "price": 1.750, "description": "Large cappuccino with steamed milk foam", "isActive": True},
    {"productId": 109, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Caramel Latte Hot M", "price": 1.750, "description": "Medium hot caramel latte", "isActive": True},
    {"productId": 110, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Caramel Latte Hot L", "price": 2.000, "description": "Large hot caramel latte", "isActive": True},
    {"productId": 111, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Hot Green Tea", "price": 1.250, "description": "Fresh brewed fragrant Japanese green tea", "isActive": True},

    # FLOAT (id: 23)
    {"productId": 112, "productCategoryId": 23, "categoryName": "FLOAT", "productName": "Americano Float", "price": 2.250, "description": "Iced americano topped with vanilla soft ice cream float", "isActive": True},
    {"productId": 113, "productCategoryId": 23, "categoryName": "FLOAT", "productName": "Cafe Latte Float", "price": 2.500, "description": "Chilled latte topped with soft ice cream", "isActive": True},
    {"productId": 114, "productCategoryId": 23, "categoryName": "FLOAT", "productName": "Caramel Latte Float", "price": 2.750, "description": "Rich caramel latte with vanilla float", "isActive": True},
    {"productId": 115, "productCategoryId": 23, "categoryName": "FLOAT", "productName": "Cola Ice Cream Float", "price": 2.000, "description": "Chilled fizzy cola topped with vanilla soft ice", "isActive": True},

    # KAKIGORI (id: 20)
    {"productId": 116, "productCategoryId": 20, "categoryName": "KAKIGORI", "productName": "Strawberry Kakigori", "price": 2.500, "description": "Fluffy shaved ice with fresh strawberry syrup and condensed milk", "isActive": True},
    {"productId": 117, "productCategoryId": 20, "categoryName": "KAKIGORI", "productName": "Mango Kakigori", "price": 2.500, "description": "Shaved ice drizzled with ripe mango puree and cream", "isActive": True},

    # SHAKES & AFFOGATO (id: 21)
    {"productId": 118, "productCategoryId": 21, "categoryName": "SHAKES & AFFOGATO", "productName": "Vanilla Milkshake", "price": 2.000, "description": "Thick creamy vanilla shake blended with rich milk", "isActive": True},
    {"productId": 119, "productCategoryId": 21, "categoryName": "SHAKES & AFFOGATO", "productName": "Chocolate Milkshake", "price": 2.000, "description": "Decadent chocolate milkshake with cocoa drizzle", "isActive": True},
    {"productId": 120, "productCategoryId": 21, "categoryName": "SHAKES & AFFOGATO", "productName": "Affogato", "price": 2.250, "description": "Hot espresso shot poured over cold vanilla ice cream", "isActive": True},

    # SOFT ICE (id: 22)
    {"productId": 121, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Vanilla Soft Ice", "price": 1.000, "description": "Creamy soft serve vanilla ice cream", "isActive": True},
    {"productId": 123, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Chocolate Soft Ice", "price": 1.000, "description": "Rich chocolate soft serve", "isActive": True},
    {"productId": 124, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Caramel Banana Soft Ice", "price": 1.750, "description": "Caramel and banana flavored soft ice cream", "isActive": True},
    {"productId": 125, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Strawberry Soft Ice", "price": 1.000, "description": "Fresh strawberry soft serve", "isActive": True},

    # OTHERS (id: 24)
    {"productId": 126, "productCategoryId": 24, "categoryName": "OTHERS", "productName": "Butter Croissant", "price": 1.500, "description": "Flaky golden fresh baked butter pastry", "isActive": True},
    {"productId": 127, "productCategoryId": 24, "categoryName": "OTHERS", "productName": "Sparkling Water", "price": 0.750, "description": "Chilled bottle of natural sparkling mineral water", "isActive": True},
]

# Add-ons available in BHD (Bahraini Dinar) - attached to specific qualifying items
ADDON_PRODUCTS = [
    {"addonId": 1, "name": "Non-Dairy Milk", "price": 0.500},
    {"addonId": 2, "name": "Extra Fruit", "price": 0.500},
    {"addonId": 3, "name": "Extra Sauce", "price": 0.500},
    {"addonId": 4, "name": "Extra Soft", "price": 0.500},
    {"addonId": 5, "name": "Extra Shot", "price": 0.500},
]

# These item names are eligible for add-ons (case-insensitive substring match)
ADDON_ELIGIBLE_ITEMS = {
    "affogato",
    "americano float",
    "americano hot l",
    "americano hot m",
    "cafe latte float",
    "cappuccino m",
    "cappuccino l",
    "cappuccino hot m",
    "cappuccino hot l",
    "caramel banana soft ice",
    "caramel latte float",
    "caramel latte hot l",
    "caramel latte hot m",
}


class MenuAPIClient:
    """Client for retrieving categories and products from REST APIs with caching and fallback."""

    def __init__(self, force_mock: bool = False):
        self.force_mock = force_mock
        self._categories_cache: Optional[List[Dict[str, Any]]] = None
        self._products_cache: Optional[List[Dict[str, Any]]] = None

    def _get_auth_headers(self) -> Dict[str, str]:
        Config.reload_env()
        token = Config.API_TOKEN.strip()
        headers = {"Content-Type": "application/json"}
        if token:
            # Bearer token header
            headers["Authorization"] = f"Bearer {token}" if not token.lower().startswith("bearer ") else token
        return headers

    def get_categories(self) -> Dict[str, Any]:
        """Fetch categories from PRODUCTS_CATEGORY_URL."""
        Config.reload_env()
        url = Config.PRODUCTS_CATEGORY_URL

        try:
            resp = requests.get(url, headers=self._get_auth_headers(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                cats = data.get("data", []) if isinstance(data, dict) else data
                # Filter active categories
                active_cats = [
                    {
                        "categoryId": c.get("productCategoryId") or c.get("id"),
                        "categoryName": c.get("categoryName") or c.get("name"),
                        "description": c.get("description", "")
                    }
                    for c in cats if c.get("isActive", True)
                ]
                self._categories_cache = active_cats
                return {"success": True, "categories": active_cats}
            else:
                logger.warning(f"Categories API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Error fetching categories from API: {e}")

        # Fallback to sample categories if API is unavailable or offline
        if Config.ENABLE_MOCK_FALLBACK or self.force_mock:
            sample_cats = [
                {"categoryId": 19, "categoryName": "COLD DRINKS", "description": "Chilled coffees, teas, and sodas"},
                {"categoryId": 18, "categoryName": "HOT DRINKS", "description": "Hot brewed coffee and tea"},
                {"categoryId": 23, "categoryName": "FLOAT", "description": "Fizzy sodas with ice cream"},
                {"categoryId": 20, "categoryName": "KAKIGORI", "description": "Japanese style shaved ice"},
                {"categoryId": 21, "categoryName": "SHAKES & AFFOGATO", "description": "Thick milkshakes and espresso affogato"},
                {"categoryId": 22, "categoryName": "SOFT ICE", "description": "Soft serve ice cream cones and cups"},
                {"categoryId": 24, "categoryName": "OTHERS", "description": "Snacks, pastries, and water"}
            ]
            return {"success": True, "categories": sample_cats, "note": "Using fallback category data"}

        return {"success": False, "error": "Unable to connect to Categories API. Please check touch screen or call staff."}

    def get_products(self, category_id: Optional[Any] = None, category_name: Optional[str] = None) -> Dict[str, Any]:
        """Fetch products from PRODUCTS_API and optionally filter by category."""
        Config.reload_env()
        url = Config.PRODUCTS_API

        if not self.force_mock:
            try:
                headers = self._get_auth_headers()
                params = {}
                if category_id:
                    params["productCategoryId"] = category_id

                resp = requests.get(url, headers=headers, params=params, timeout=5)
                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_products = res_json.get("data", []) if isinstance(res_json, dict) else res_json
                    if isinstance(raw_products, dict):
                        raw_products = raw_products.get("products") or raw_products.get("items") or []

                    products = []
                    for p in raw_products:
                        # Only offer SELLABLE customer items (skip RAW_MATERIAL inventory like cups and lids)
                        if p.get("itemType") and p.get("itemType") != "SELLABLE":
                            continue
                        if not p.get("isActive", True):
                            continue

                        cat_obj = p.get("category") or {}
                        c_name = cat_obj.get("categoryName", "") if isinstance(cat_obj, dict) else ""
                        cat_id = p.get("productCategoryId") or p.get("categoryId")
                        
                        # Apply category filter if given
                        if category_id and str(cat_id) != str(category_id):
                            continue
                        if category_name and category_name.lower() not in c_name.lower():
                            continue

                        # Extract price: check finalPrice, basePrice, price, unitPrice
                        price_val = p.get("finalPrice") or p.get("basePrice") or p.get("price") or p.get("unitPrice") or 0.0
                        try:
                            price = float(price_val)
                        except (ValueError, TypeError):
                            price = 0.0

                        p_name = p.get("productName") or p.get("name") or ""
                        is_eligible = self.is_addon_eligible(p_name)
                        addons_list = []
                        if is_eligible:
                            # Extract available add-ons from API modifierGroups
                            for mg in p.get("modifierGroups", []):
                                for opt in mg.get("options", []):
                                    if opt.get("isAvailable", True):
                                        try:
                                            adj = float(opt.get("priceAdjustment", 0.5))
                                        except (ValueError, TypeError):
                                            adj = 0.5
                                        addons_list.append({
                                            "addonId": opt.get("modifierOptionId"),
                                            "name": opt.get("name"),
                                            "price": round(adj, 3)
                                        })
                            if not addons_list:
                                addons_list = list(ADDON_PRODUCTS)

                        products.append({
                            "productId": p.get("productId") or p.get("id"),
                            "name": p_name,
                            "price": round(price, 3),
                            "category": c_name or f"Category {cat_id}",
                            "description": p.get("description", "") or "",
                            "available": True,
                            "addon_eligible": is_eligible,
                            "addons": addons_list
                        })
                    self._products_cache = products
                    return {"success": True, "products": products}
                elif resp.status_code == 401:
                    logger.warning(f"Products API Unauthorized (Token Expired or Invalid): {resp.text}")
                    if not Config.ENABLE_MOCK_FALLBACK:
                        return {
                            "success": False,
                            "error": "Products API Unauthorized: Token in .env.development is invalid or expired. Please check touch screen or call staff."
                        }
            except Exception as e:
                logger.error(f"Error fetching products from API: {e}")
                if not Config.ENABLE_MOCK_FALLBACK:
                    return {"success": False, "error": f"API error: {e}. Please check touch screen or call staff."}

        # Fallback to realistic mock menu
        products = []
        for p in SAMPLE_PRODUCTS:
            if not p.get("isActive", True):
                continue
            cat_id = p.get("productCategoryId")
            c_name = p.get("categoryName", "")

            if category_id and str(cat_id) != str(category_id):
                continue
            if category_name and category_name.lower() not in c_name.lower():
                continue

            p_name = p["productName"]
            is_eligible = self.is_addon_eligible(p_name)
            addons_list = list(ADDON_PRODUCTS) if is_eligible else []

            products.append({
                "productId": p["productId"],
                "name": p_name,
                "price": round(float(p["price"]), 3),
                "category": c_name,
                "description": p.get("description", ""),
                "available": True,
                "addon_eligible": is_eligible,
                "addons": addons_list
            })

        return {"success": True, "products": products}

    def search_product(self, query: str) -> Dict[str, Any]:
        """Search products by customer query or keyword with relevance scoring."""
        import re
        q = query.strip().lower()
        if not q:
            return {"success": True, "found": False, "products": []}
        
        # Ensure products are loaded
        res = self.get_products()
        if not res.get("success"):
            return res

        all_products = res.get("products", [])
        q_words = [w for w in re.findall(r'\w+', q) if len(w) > 2]
        scored_products = []

        for p in all_products:
            score = 0
            name = p["name"].lower()
            cat = p.get("category", "").lower()
            desc = p.get("description", "").lower()

            if q == name:
                score = 150
            elif q == cat:
                score = 120
            elif q in name:
                score += 70
            elif q in cat or cat in q:
                score += 60
            elif q in desc:
                score += 35

            # Word boundary matching
            for w in q_words:
                if re.search(r'\b' + re.escape(w) + r'\b', name):
                    score += 25
                elif re.search(r'\b' + re.escape(w) + r'\b', cat):
                    score += 25
                elif re.search(r'\b' + re.escape(w) + r'\b', desc):
                    score += 15

            if score > 0:
                scored_products.append((score, p))

        # Sort products by score descending
        scored_products.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in scored_products]

        if not results:
            return {
                "success": True,
                "found": False,
                "query": query,
                "products": [],
                "message": f"No available product found matching '{query}'."
            }

        # Return at most 3 options as per prompt rule: 'mention at most 3 and ask which one'
        return {
            "success": True,
            "found": True,
            "query": query,
            "products": results[:3],
            "total_matches": len(results),
            "hint": f"Available options matching '{query}'. If customer asked to order or have this, immediately call add_item(product_name='{results[0]['name']}', quantity=1). Otherwise if they are just browsing, mention at most 3 options."
        }

    def get_addons(self, item_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the list of available add-ons with their prices."""
        if item_name and not self.is_addon_eligible(item_name):
            return {
                "success": False,
                "addon_eligible": False,
                "message": f"Add-ons are not available for '{item_name}'. Add-ons are only for select beverages and soft ice."
            }
        return {
            "success": True,
            "addon_eligible": True,
            "addons": ADDON_PRODUCTS,
            "note": "Each add-on costs 0.500 BHD. Ask the customer which ones they want."
        }

    def is_addon_eligible(self, item_name: str) -> bool:
        """Check whether an item name is eligible to receive add-ons.
        Eligible products (only these 11):
          1. Affogato (plain, not Matcha Affogato)
          2. Americano Float
          3. Americano Hot L
          4. Americano Hot M
          5. Cafe Latte Float
          6. Cappuccino Hot M / Cappuccino M
          7. Cappuccino Hot L / Cappuccino L
          8. Caramel Banana Soft Ice
          9. Caramel Latte Float / Caramel latte Float
          10. Caramel Latte Hot L
          11. Caramel Latte Hot M
        """
        if not item_name:
            return False
        name = item_name.strip().lower()

        if "matcha" in name:
            return False

        if "affogato" in name:
            return True
        if "americano float" in name:
            return True
        if "americano hot" in name and (" l" in name or name.endswith("l") or "large" in name):
            return True
        if "americano hot" in name and (" m" in name or name.endswith("m") or "medium" in name):
            return True
        if "cafe latte float" in name:
            return True
        if "cappuccino" in name and (" l" in name or name.endswith("l") or "large" in name):
            return True
        if "cappuccino" in name and (" m" in name or name.endswith("m") or "medium" in name):
            return True
        if "caramel banana soft ice" in name:
            return True
        if "caramel latte float" in name:
            return True
        if "caramel latte hot" in name and (" l" in name or name.endswith("l") or "large" in name):
            return True
        if "caramel latte hot" in name and (" m" in name or name.endswith("m") or "medium" in name):
            return True

        return False
