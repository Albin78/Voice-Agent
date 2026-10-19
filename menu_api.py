import requests
import json
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger("MenuAPI")

# Realistic fallback menu based on the live categories (IDs 18-24)
SAMPLE_PRODUCTS = [
    # COLD DRINKS (id: 19)
    {"productId": 101, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Iced Americano", "price": 3.50, "description": "Bold espresso over ice and cold water", "isActive": True},
    {"productId": 102, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Iced Caramel Latte", "price": 4.75, "description": "Rich espresso with chilled milk and caramel syrup", "isActive": True},
    {"productId": 103, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Cold Brew Coffee", "price": 4.25, "description": "Slow-steeped smooth iced coffee", "isActive": True},
    {"productId": 104, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Iced Lemon Tea", "price": 3.00, "description": "Refreshing black tea with fresh lemon slices", "isActive": True},
    {"productId": 122, "productCategoryId": 19, "categoryName": "COLD DRINKS", "productName": "Fresh Apple Juice", "price": 3.75, "description": "Freshly pressed chilled apple juice", "isActive": True},
    
    # HOT DRINKS (id: 18)
    {"productId": 105, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Hot Cappuccino", "price": 4.00, "description": "Steamed milk foam over velvety espresso", "isActive": True},
    {"productId": 106, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Hot Latte", "price": 4.25, "description": "Smooth espresso with steamed whole milk", "isActive": True},
    {"productId": 107, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Espresso Single", "price": 2.50, "description": "Classic concentrated single shot of espresso", "isActive": True},
    {"productId": 108, "productCategoryId": 18, "categoryName": "HOT DRINKS", "productName": "Hot Green Tea", "price": 3.00, "description": "Fresh brewed fragrant Japanese green tea", "isActive": True},

    # FLOAT (id: 23)
    {"productId": 109, "productCategoryId": 23, "categoryName": "FLOAT", "productName": "Cola Ice Cream Float", "price": 4.50, "description": "Chilled fizzy cola topped with vanilla soft ice", "isActive": True},
    {"productId": 110, "productCategoryId": 23, "categoryName": "FLOAT", "productName": "Matcha Float", "price": 5.25, "description": "Iced ceremonial matcha latte topped with soft ice cream", "isActive": True},

    # KAKIGORI (id: 20)
    {"productId": 111, "productCategoryId": 20, "categoryName": "KAKIGORI", "productName": "Strawberry Kakigori", "price": 6.00, "description": "Fluffy shaved ice with fresh strawberry syrup and condensed milk", "isActive": True},
    {"productId": 112, "productCategoryId": 20, "categoryName": "KAKIGORI", "productName": "Mango Kakigori", "price": 6.50, "description": "Shaved ice drizzled with ripe mango puree and cream", "isActive": True},

    # SHAKES & AFFOGATO (id: 21)
    {"productId": 113, "productCategoryId": 21, "categoryName": "SHAKES & AFFOGATO", "productName": "Vanilla Milkshake", "price": 5.00, "description": "Thick creamy vanilla shake blended with rich milk", "isActive": True},
    {"productId": 114, "productCategoryId": 21, "categoryName": "SHAKES & AFFOGATO", "productName": "Chocolate Milkshake", "price": 5.25, "description": "Decadent chocolate milkshake with cocoa drizzle", "isActive": True},
    {"productId": 115, "productCategoryId": 21, "categoryName": "SHAKES & AFFOGATO", "productName": "Classic Affogato", "price": 4.50, "description": "Hot espresso shot poured over cold vanilla ice cream", "isActive": True},

    # SOFT ICE (id: 22)
    {"productId": 116, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Vanilla Soft Ice Cone", "price": 3.00, "description": "Creamy soft serve vanilla ice cream in a crisp waffle cone", "isActive": True},
    {"productId": 117, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Chocolate Soft Ice Cone", "price": 3.25, "description": "Rich dark chocolate soft serve in waffle cone", "isActive": True},
    {"productId": 118, "productCategoryId": 22, "categoryName": "SOFT ICE", "productName": "Twist Soft Ice Cup", "price": 3.50, "description": "Vanilla and chocolate swirl served in a cup", "isActive": True},

    # OTHERS (id: 24)
    {"productId": 119, "productCategoryId": 24, "categoryName": "OTHERS", "productName": "Butter Croissant", "price": 3.25, "description": "Flaky golden fresh baked butter pastry", "isActive": True},
    {"productId": 120, "productCategoryId": 24, "categoryName": "OTHERS", "productName": "Chocolate Chip Muffin", "price": 3.00, "description": "Warm fluffy muffin loaded with chocolate chips", "isActive": True},
    {"productId": 121, "productCategoryId": 24, "categoryName": "OTHERS", "productName": "Sparkling Water", "price": 2.25, "description": "Chilled bottle of natural sparkling mineral water", "isActive": True}
]


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

                        products.append({
                            "productId": p.get("productId") or p.get("id"),
                            "name": p.get("productName") or p.get("name"),
                            "price": round(price, 2),
                            "category": c_name or f"Category {cat_id}",
                            "description": p.get("description", "") or "",
                            "available": True
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

            products.append({
                "productId": p["productId"],
                "name": p["productName"],
                "price": float(p["price"]),
                "category": c_name,
                "description": p.get("description", ""),
                "available": True
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
