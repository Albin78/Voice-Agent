import ast
import requests
import json
import logging
import random
import re
from typing import Dict, Any, List, Optional
from config import Config
from cart import OrderCart
from menu_api import MenuAPIClient
from tools import TOOL_DEFINITIONS, ToolDispatcher
from tts import clean_spoken_text
from stt import is_repetition_hallucination

logger = logging.getLogger("ChefAgent")


TOOL_NAME_MAP = {
    "additem": "add_item",
    "add_item": "add_item",
    "removeitem": "remove_item",
    "remove_item": "remove_item",
    "update_item_notes": "update_item_notes",
    "updateitemnotes": "update_item_notes",
    "searchproduct": "search_product",
    "search_product": "search_product",
    "getproducts": "get_products",
    "get_products": "get_products",
    "getcategories": "get_categories",
    "get_categories": "get_categories",
    "getcart": "get_cart",
    "get_cart": "get_cart",
    "confirm_and_submit": "confirm_and_submit",
    "confirmandsubmit": "confirm_and_submit",
    "recommend_by_mood": "recommend_by_mood",
    "recommendbymood": "recommend_by_mood",
    "get_addons": "get_addons",
    "getaddons": "get_addons",
    "add_addon": "add_addon",
    "addaddon": "add_addon",
    "clearcart": "clear_cart",
    "clear_cart": "clear_cart",
    "emptycart": "clear_cart",
    "empty_cart": "clear_cart",
    "call_staff": "call_staff",
    "callstaff": "call_staff",
    "update_quantity": "update_quantity",
    "updatequantity": "update_quantity",
    "set_quantity": "update_quantity",
    "setquantity": "update_quantity",
    "add_more": "add_more",
    "addmore": "add_more",
}


def _extract_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
    """
    Extract function calls if a local model outputs them in text rather than the tool_calls field.
    Handles CLI syntax (`additem "green tea" 1 {"notes": "..."}`), Python syntax (`add_item(...)`),
    and JSON syntax (`add_item {"product_name": ...}`).
    """
    if not content:
        return []

    tool_calls = []

    # 1. CLI syntax: additem "name" [qty] [{"notes": ...}]
    cli_add_re = re.search(r'\b(add_?item)\s+["\']([^"\']+)["\'](?:\s+(\d+))?(?:\s*(\{.*?\}))?', content, re.IGNORECASE | re.DOTALL)
    if cli_add_re:
        p_name = cli_add_re.group(2).strip()
        qty = int(cli_add_re.group(3)) if cli_add_re.group(3) else 1
        notes = ""
        if cli_add_re.group(4):
            try:
                extra = json.loads(cli_add_re.group(4))
                notes = extra.get("notes", "")
            except Exception:
                nm = re.search(r'["\']notes["\']\s*:\s*["\']([^"\']*)["\']', cli_add_re.group(4))
                if nm:
                    notes = nm.group(1)
        args = {"product_name": p_name, "quantity": qty}
        if notes:
            args["notes"] = notes
        return [{"function": {"name": "add_item", "arguments": args}}]

    # 1b. CLI remove syntax: removeitem "name" [qty]
    cli_rem_re = re.search(r'\b(remove_?item)\s+["\']([^"\']+)["\'](?:\s+(\d+))?', content, re.IGNORECASE)
    if cli_rem_re:
        p_name = cli_rem_re.group(2).strip()
        qty = int(cli_rem_re.group(3)) if cli_rem_re.group(3) else 1
        return [{"function": {"name": "remove_item", "arguments": {"product_name": p_name, "quantity": qty}}}]

    # 1c. CLI mood recommendation syntax: recommend_by_mood "mood"
    cli_rec_re = re.search(r'\b(recommend_?by_?mood)\s+["\']([^"\']+)["\']', content, re.IGNORECASE)
    if cli_rec_re:
        mood = cli_rec_re.group(2).strip()
        return [{"function": {"name": "recommend_by_mood", "arguments": {"mood": mood}}}]

    # 2. Check Python call syntax: fn_name(...)
    known_keys = list(TOOL_NAME_MAP.keys())
    py_pattern = rf'\b({"|".join(known_keys)})\s*\((.*?)\)'
    for m in re.finditer(py_pattern, content, flags=re.DOTALL | re.IGNORECASE):
        raw_name = m.group(1).lower()
        fn_name = TOOL_NAME_MAP.get(raw_name, raw_name)
        arg_str = m.group(2).strip()
        args = {}
        try:
            tree = ast.parse(f"{fn_name}({arg_str})")
            call_node = tree.body[0].value
            for kw in call_node.keywords:
                args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            for kv in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([0-9.]+)|(\w+))', arg_str):
                k = kv.group(1)
                v = kv.group(2) or kv.group(3) or kv.group(4) or kv.group(5)
                try:
                    v = float(v) if '.' in v else int(v)
                except Exception:
                    pass
                args[k] = v
        tool_calls.append({"function": {"name": fn_name, "arguments": args}})

    if tool_calls:
        return tool_calls

    # 3. Check JSON syntax: fn_name {...}
    json_pattern = rf'\b({"|".join(known_keys)})\s*(\{{.*?\}})'
    for m in re.finditer(json_pattern, content, flags=re.DOTALL | re.IGNORECASE):
        raw_name = m.group(1).lower()
        fn_name = TOOL_NAME_MAP.get(raw_name, raw_name)
        arg_str = m.group(2).strip()
        try:
            args = json.loads(arg_str)
            tool_calls.append({"function": {"name": fn_name, "arguments": args}})
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    # 4. Spoken declaration fallback: e.g. "Adding an Affogato to your order...", "Removing the Affogato..."
    # Clear cart spoken declaration fallback
    if re.search(r'\b(?:clearing|cleared|empty|emptied|i\'ve cleared|i have cleared|removing everything|removed everything|remove everything|clear everything)\b', content, re.IGNORECASE):
        return [{"function": {"name": "clear_cart", "arguments": {}}}]

    # Remove item spoken declaration fallback
    rem_match = re.search(r'\b(?:removing|removed|i\'ve removed|i have removed|take off|taking off|cancel|cancelled)\s+(?:the\s+|my\s+|a\s+|an\s+|one\s+)?([A-Za-z0-9 ]+?)(?:\s+from your (?:order|cart)|\s+from (?:order|cart)|\.|\,|$)', content, re.IGNORECASE)
    if rem_match:
        cand_name = rem_match.group(1).strip()
        if len(cand_name) > 2 and cand_name.lower() not in {"this", "that", "it", "item"}:
            if cand_name.lower() in {"everything", "all", "all items", "all the items", "cart", "the cart", "my cart"}:
                return [{"function": {"name": "clear_cart", "arguments": {}}}]
            return [{"function": {"name": "remove_item", "arguments": {"product_name": cand_name, "quantity": 1}}}]

    add_match = re.search(r'\b(?:adding|added|i\'ve added|i have added)\s+(?:an?|one|\d+)?\s*([A-Za-z0-9 ]+?)(?:\s+to your order|\s+to your cart|\.|\,)', content, re.IGNORECASE)
    if add_match:
        cand_name = add_match.group(1).strip()
        if len(cand_name) > 2 and cand_name.lower() not in {"this", "that", "it", "item"}:
            tool_calls.append({"function": {"name": "add_item", "arguments": {"product_name": cand_name, "quantity": 1}}})

    return tool_calls


def format_friendly_reply_if_code(text: str, cart: OrderCart) -> str:
    """
    Guarantees that Chef NEVER speaks raw code, function names, brackets, or CLI syntax aloud.
    Transforms accidental code leakage into warm, professional kiosk dialogue.
    """
    if not text:
        return "I'm ready to help with your order — what can I get for you?"

    code_indicators = [
        r'\b(?:add_?item|remove_?item|search_?product|recommend_?by_?mood|call_?staff|update_?item_?notes)\b',
        r'[{\[].*?["\':].*?[}\]]',
        r'["\']notes["\']\s*:',
        r'^[│|\s]*add_?item',
    ]

    has_code = any(re.search(pat, text, flags=re.IGNORECASE) for pat in code_indicators)
    if not has_code:
        return text

    # Transform leaked tool call into warm customer dialogue
    m_add = re.search(r'\badd_?item\s*(?:\(|["\'])\s*([^"\'(){\n]+)', text, re.IGNORECASE)
    notes_m = re.search(r'["\']notes["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    notes_str = f" with {notes_m.group(1)}" if notes_m else ""

    if m_add:
        item_raw = m_add.group(1).strip().strip('"\'')
        item_clean = item_raw.title()
        return f"I've added {item_clean}{notes_str} to your order. Would you like anything else?"

    m_rem = re.search(r'\bremove_?item\s*(?:\(|["\'])\s*([^"\'(){\n]+)', text, re.IGNORECASE)
    if m_rem:
        item_raw = m_rem.group(1).strip().strip('"\'')
        item_clean = item_raw.title()
        return f"I've removed {item_clean} from your order. What else can I get for you?"

    if "recommend_by_mood" in text.lower() or "recommendbymood" in text.lower():
        return "I have some wonderful recommendations tailored for your mood! What sounds good to you?"

    clean = re.sub(r'[{}\[\]"\'`|│]', '', text)
    clean = re.sub(r'\b(add_?item|remove_?item|get_?cart|call_?staff)\b', '', clean, flags=re.IGNORECASE).strip()
    if len(clean) > 5:
        return clean

    return "I've updated your order. What else can I get for you today?"


def condense_long_spoken_list(text: str) -> str:
    """
    If an agent response enumerates more than 3 items with prices in a single speech turn,
    condense it to the top 3 items to preserve voice clarity and prevent customer cognitive overload.
    Example: 9 cold drinks enumerated -> Top 3 + 'plus more options on the screen. Which one would you like?'
    """
    if not text:
        return text

    pattern = re.compile(
        r'(?:(?:we\s+(?:also\s+)?have|options\s+include|like|and)\s+)?'
        r'([A-Za-z0-9\s\-]+?)\s+(?:for|at)\s+(\d+(?:\.\d+)?)\s*BHD',
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    if len(matches) <= 3:
        return text

    intro_match = re.match(
        r'^(.*?\b(?:we\s+(?:also\s+)?have|choices\s+include|options\s+are|popular\s+choices\s+include)\b)',
        text,
        re.IGNORECASE
    )
    if intro_match:
        intro = intro_match.group(1).strip() + " "
    else:
        intro = "We also have "

    top_3 = matches[:3]
    parts = []
    for m in top_3:
        item_name = m.group(1).strip()
        item_name = re.sub(r'^(?:and\s+|we\s+(?:also\s+)?have\s+)', '', item_name, flags=re.IGNORECASE).strip()
        price = m.group(2)
        parts.append(f"{item_name} for {price} BHD")

    top_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    last_end = matches[-1].end()
    trailing = text[last_end:].strip()
    q_match = re.search(r'([A-Z][^.!?]*\?)$', trailing)
    question = q_match.group(1) if q_match else "Which one would you like?"

    return f"{intro}{top_str}, plus more options on the screen. {question}"

# ---------------------------------------------------------------------------
# Pre-LLM pattern matching — evaluated BEFORE sending to the model.
# These catch common cases instantly, saving a full LLM round-trip.
# ---------------------------------------------------------------------------

# Customer explicitly asks to speak with a human
_CALL_STAFF_RE = re.compile(
    r'\b('
    r'speak (to|with) (a )?(person|human|staff|manager|someone)|'
    r'call (the )?staff|get me (a )?manager|'
    r'i want (to )?talk (to|with) (a )?(person|human)|'
    r'human please|real person|get someone|someone else'
    r')\b',
    re.IGNORECASE,
)

# Identity / meta questions about Chef itself
_IDENTITY_RE = re.compile(
    r'\b('
    r'what is your name|who are you|'
    r'are you (a )?(robot|bot|ai|machine|computer|human|real)|'
    r'are you alive|how do you work|what can you do|'
    r'are you a (person|program)|what are you'
    r')\b',
    re.IGNORECASE,
)

# Clearly off-topic signals — things definitively unrelated to a restaurant
_OFF_TOPIC_RE = re.compile(
    r'\b('
    r'weather|forecast|raining|sunny|temperature outside|hot today|cold today|'
    r'news|politics|election|president|prime minister|parliament|war|crisis|conflict|'
    r'stocks?|bitcoin|crypto|nft|invest(ment)?|'
    r'tell me a joke|sing (a |me a )?song|play music|dance for me|'
    r'fix (my )?(phone|laptop|computer|car|internet)|tech support|help me install|'
    r'i love you|will you marry me|date me|be my girlfriend|'
    r'what is the meaning of life|philosophy class|debate|'
    r'sports? (score|result|who won)|latest match|football result|'
    r'translate (this|for me)|grammar check|write (my|an?) essay'
    r')\b',
    re.IGNORECASE,
)

# Restaurant-topic safety net — if ANY of these appear, suppress off-topic block
_RESTAURANT_RE = re.compile(
    r'\b('
    r'menu|order|eat|drink|food|coffee|tea|juice|water|hungry|thirsty|'
    r'item|price|total|bill|cart|buy|get me|give me|i want|i would like|'
    r'add|remove|cancel|checkout|pay|confirm|'
    r'burger|pizza|sandwich|salad|dessert|'
    r'ice cream|milkshake|shake|espresso|latte|cappuccino|americano|'
    r'cold brew|affogato|kakigori|soft ice|float|croissant|muffin|'
    r'staff|manager|allergy|ingredient|halal|vegan|gluten|calorie|nut|'
    r'sugar|sweet|ice|hot|cold|mood|tired|stress|relax|energy|refresh|recommend|healthy|light'
    r')\b',
    re.IGNORECASE,
)

# Clear whole cart / remove everything pattern
_CLEAR_CART_RE = re.compile(
    r'\b('
    r'(?:clear|empty|remove|delete|cancel)\s+(?:all|everything|all\s+the\s+items|all\s+items|the\s+entire\s+cart|the\s+whole\s+cart|my\s+cart|the\s+cart)(?:\s+from\s+(?:my\s+|the\s+)?cart)?|'
    r'clear\s+cart|empty\s+cart|start\s+over|reset\s+order'
    r')\b',
    re.IGNORECASE
)

# Remove specific item pattern
_REMOVE_ITEM_RE = re.compile(
    r'^(?:please\s+)?(?:remove|take\s+off|cancel|delete)\s+(.+?)(?:\s+from\s+(?:my\s+|the\s+)?(?:cart|order))?$',
    re.IGNORECASE
)

# Ordering / Add item intent pattern — catches explicit ordering speech
_ADD_ITEM_INTENT_RE = re.compile(
    r'^(?:please\s+)?'
    r'(?:'
    r'add|'
    r'can\s+(?:i|you)\s+(?:get|have|add)|'
    r'i\s+(?:want|would\s+like|\x27d\s+like|\x27ll\s+take|need|\x27m\s+going\s+to\s+have|am\s+going\s+to\s+have|\x27ll\s+go\s+with|\x27m\s+going\s+for)|'
    r'give\s+me|'
    r'get\s+me|'
    r'(?:i\x27ll|i\s+will)\s+(?:have|take|get|go\s+with)'
    r')\s+'
    r'(.+?)'
    r'(?:\s+(?:to|in|into)\s+(?:my\s+|the\s+)?(?:cart|order))?$',
    re.IGNORECASE
)

# Menu / Category options requests
_OPTIONS_QUERY_RE = re.compile(
    r'\b('
    r'options?(?:\s+(?:of|for|in|available))?|'
    r'get\s+me\s+(?:the\s+)?options|'
    r'give\s+me\s+(?:the\s+)?options|'
    r'what\s+are\s+(?:the\s+)?(?:other\s+)?(?:available\s+)?(?:hot|cold|soft\s+ice|float|shake|kakigori)?\s*(?:drinks?|items?|beverages?|coffees?|options?)(?:\s+(?:available|there))?|'
    r'what\s+(?:other\s+)?(?:hot|cold|soft\s+ice|float|shake|kakigori)?\s*(?:drinks?|items?|beverages?|coffees?|options?)\s*(?:are\s+available|do\s+you\s+have)|'
    r'what\s+are\s+(?:the\s+)?options|'
    r'show\s+(?:me\s+)?(?:the\s+)?options|'
    r'tell\s+(?:me\s+)?(?:the\s+)?options|'
    r'list\s+(?:the\s+)?(?:options|items)|'
    r'what\s+(?:kinds?\s+of\s+|types?\s+of\s+)?(?:hot|cold|soft\s+ice|drinks?|coffees?|food|dessert|float|shake|kakigori).*?do\s+you\s+have|'
    r'what\s+(?:do\s+you\s+have|are\s+your)\s+in\b|'
    r'what\s+do\s+you\s+have\s+for\b|'
    r'what\s+else\s+do\s+you\s+have|'
    r'see\s+all\s+items|view\s+all\s+items|list\s+all\s+items|'
    r'what\s+(?:hot|cold)\s+drinks\s+do\s+you\s+have|'
    r'show\s+me\s+(?:all\s+)?(?:the\s+)?(?:hot\s+drinks|cold\s+drinks|soft\s+ice|floats?|shakes?|kakigori|menu)|'
    r'show\s+me\s+(?:your\s+)?menu|what(?:\x27s|\s+is)\s+on\s+the\s+menu|can\s+i\s+see\s+the\s+menu|view\s+the\s+menu'
    r')\b',
    re.IGNORECASE
)

_CATEGORY_MAP = [
    ("HOT DRINKS", re.compile(r'\b(?:hot\s+drinks?|hot\s+coffees?|hot\s+beverages?|hot\s+tea)\b', re.IGNORECASE)),
    ("COLD DRINKS", re.compile(r'\b(?:cold\s+drinks?|iced?\s+drinks?|cold\s+coffees?|iced?\s+coffees?|cold\s+brew|cold\s+beverages?)\b', re.IGNORECASE)),
    ("SOFT ICE", re.compile(r'\b(?:soft\s*ice|soft\s*serve|ice\s*cream)\b', re.IGNORECASE)),
    ("FLOAT", re.compile(r'\b(?:floats?|ice\s*cream\s*floats?)\b', re.IGNORECASE)),
    ("KAKIGORI", re.compile(r'\b(?:kakigori|shaved\s*ice)\b', re.IGNORECASE)),
    ("SHAKES & AFFOGATO", re.compile(r'\b(?:shakes?|milkshakes?|affogato)\b', re.IGNORECASE)),
    ("OTHERS", re.compile(r'\b(?:croissants?|pastr(?:y|ies)|snacks?|others?)\b', re.IGNORECASE)),
]

# Order checkout and placement pattern
_PLACE_ORDER_RE = re.compile(
    r'\b('
    r'place\s+(?:the\s+|my\s+|this\s+|that\s+)?order|'
    r'confirm\s+(?:the\s+|my\s+|this\s+|that\s+)?order|'
    r'submit\s+(?:the\s+|my\s+|this\s+|that\s+)?order|'
    r'checkout|finalize\s+(?:the\s+|my\s+|this\s+|that\s+)?order|'
    r'complete\s+(?:the\s+|my\s+|this\s+|that\s+)?order|'
    r'that(?:\x27s|\s+is)\s+all(?:,\s*place\s+(?:the\s+|my\s+|this\s+|that\s+)?order)?|'
    r'check\s+please|i(?:\x27m|\s+am)\s+done(?:\s+ordering)?|'
    r'order\s+please|place\s+it\b|order\s+now\b'
    r')\b',
    re.IGNORECASE
)

# Confirmation response patterns (when agent asked "Shall I place this order?")
_CONFIRMATION_YES_RE = re.compile(
    r'^(?:yes|yeah|yep|sure|ok|okay|please|confirm|confirm it|confirm order|'
    r'place it|place the order|place this order|place that order|place order|'
    r'go ahead|proceed|do it|sounds good|yes please|yes place it|yes place the order|'
    r'yes place this order|yes place that order|that(?:\x27s|\s+is)\s+(?:right|correct|fine|good)|'
    r'correct|all good|perfect)(?:[\s.!,].*)?$',
    re.IGNORECASE
)

_CONFIRMATION_NO_RE = re.compile(
    r'^(?:no|nope|wait|cancel|not yet|don\x27t|hold on|change)(?:[\s.!,].*)?$',
    re.IGNORECASE
)

# Direct customization pattern (sweetness, ice, temperature adjustments)
_CUSTOMIZATION_RE = re.compile(
    r'^(?:make it |with |can i get |please )?'
    r'(?:less sugar|no sugar|extra sugar|without sugar|low sugar|half sweet|normal sugar|'
    r'condensed sugar|shooting condensed sugar|shoot in sugar|'
    r'extra ice|no ice|less ice|more ice|hot|cold)\b',
    re.IGNORECASE,
)

# Warm off-topic redirect replies — rotated to avoid sounding robotic
_OFF_TOPIC_REPLIES = [
    "I'm only able to help with your food and drink order today — what can I get for you?",
    "That's a little outside my specialty, but I'd love to help you order something! What sounds good?",
    "I'm Chef, your ordering assistant here — want me to help you find something delicious?",
]

# Off-topic escalation (after 2 failed redirects)
_OFF_TOPIC_ESCALATE = (
    "I'm having trouble helping with that. Would you like to use the touch screen, "
    "or shall I get a staff member for you?"
)

# Identity replies — warm and quick
_IDENTITY_REPLIES = [
    "I'm Chef, your friendly voice ordering assistant here at {name}! What can I get for you today?",
    "I'm an AI ordering assistant called Chef, here to help you at {name}! Ready to order?",
]


class ChefAgent:
    """
    Chef — Friendly Voice Ordering Assistant at a restaurant kiosk.

    Turn flow:
      1. Silent / empty       → misunderstand handler → call_staff after 2 strikes
      2. Simple greeting       → instant warm reply (no LLM)
      3. Call-staff request    → immediately dispatch call_staff (no LLM)
      4. Direct customization  → instant cart note update (no LLM)
      5. Identity question     → instant self-intro (no LLM)
      6. Off-topic utterance   → warm redirect (no LLM); escalate after 2 strikes
      7. Restaurant utterance  → Ollama tool-calling loop
    """

    def __init__(self, cart: Optional[OrderCart] = None, menu_client: Optional[MenuAPIClient] = None):
        self.cart = cart or OrderCart()
        self.menu_client = menu_client or MenuAPIClient()
        self.dispatcher = ToolDispatcher(self.menu_client, self.cart)
        self.model = Config.OLLAMA_MODEL
        self.base_url = Config.OLLAMA_BASE_URL.rstrip('/')
        self.restaurant_name = Config.RESTAURANT_NAME

        # Dialogue state tracking
        self.misunderstand_count: int = 0
        self.off_topic_count: int = 0
        self.welcome_given: bool = False
        self.last_offered_suggestion: Optional[str] = None
        self.pending_item_to_add: Optional[Dict[str, Any]] = None
        self.messages: List[Dict[str, Any]] = []

        self._init_system_prompt()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _init_system_prompt(self):
        system_content = f"""You are "Chef", the friendly voice ordering assistant at {self.restaurant_name} restaurant kiosk.
Everything you say is spoken aloud through a speaker, so NEVER use lists, bullet points, numbers (like 1. or 2.), symbols, or markdown.
Keep all spoken replies warm, natural, and AT MOST 2 SHORT SENTENCES.

MANDATORY INTENT DISAMBIGUATION RULES (CRITICAL):
1. BROWSING / VIEWING INTENT:
   - When the customer asks to SEE, SHOW, VIEW, LIST, or BROWSE items, or asks for options (e.g. "get me the options of hot drinks", "what are the options of cold drinks", "I want to see all items in soft ice", "show me what drinks you have", "what items are in float?", "can I see the menu?", "what do you have?"):
     --> You MUST call `get_products(category_name=...)` or `get_categories()`.
     --> NEVER reply with just 'Sure', 'Of course', or conversational filler without listing the items! You MUST warmly list up to 3 options with their BHD prices in your response.
     --> NEVER call `add_item` on browsing requests! Browsing requests are NOT orders.

2. ORDERING / PURCHASING INTENT:
   - When the customer clearly asks to BUY, HAVE, GET, or ADD an item (e.g. "I want a vanilla soft ice", "give me one latte", "add hot green tea", "two coffees", "a croissant"):
     --> First call `search_product(query=<item name>)` to look up the exact catalog name.
     --> If the search result says 'EXACT MATCH FOUND', call `add_item` immediately with the exact catalog product_name.
     --> If the search result says 'FUZZY MATCH' or 'needs_confirmation', do NOT call add_item. Instead ask the customer:
         "Did you mean <product name> for <price> BHD? Shall I add that to your cart?"
     --> Only call `add_item` AFTER the customer confirms with yes/sure/go ahead/please.
     --> CRITICAL: NEVER add more than ONE item per customer request. NEVER add multiple products at once.
     --> NEVER say or claim you added an item in conversational text without calling `add_item` first!

CONTRASTIVE EXAMPLES FOR CLARITY:
- Customer: "get me the options of hot drinks" -> BROWSING -> Call `get_products(category_name="HOT DRINKS")`. List items with BHD prices! NEVER just say 'Sure'!
- Customer: "I want to see all items in soft ice" -> BROWSING -> Call `get_products(category_name="SOFT ICE")`. DO NOT call add_item!
- Customer: "I want a Vanilla Soft Ice" -> ORDERING -> Call `add_item(product_name="Vanilla Soft Ice", quantity=1)`.
- Customer: "Show me the drinks" -> BROWSING -> Call `get_categories()`.
- Customer: "Can I get one Cappuccino M" -> ORDERING -> Call `add_item(product_name="Cappuccino M", quantity=1)`.
- Customer: "remove the Affogato from my cart" -> REMOVE ITEM -> Call `remove_item(product_name="Affogato")`.
- Customer: "remove everything from cart" -> CLEAR CART -> Call `clear_cart()`.

3. REMOVING / CANCELING ITEMS (CRITICAL):
   - When the customer asks to remove, cancel, or take off an item (e.g. "remove the Affogato from my cart", "take off the latte"):
     --> You MUST call the `remove_item` tool.
   - When the customer asks to remove everything, clear, or empty the cart (e.g. "remove everything from cart", "clear my cart", "empty the cart"):
     --> You MUST call the `clear_cart` tool.
   - NEVER say or claim you removed an item or cleared the cart in text without calling the tool first!

4. CUSTOMIZING ITEMS (SWEETNESS / ICE / NOTES):
   - When the customer specifies sweetness or customization for an item (e.g. "less sugar", "no sugar", "extra ice", "oat milk"), call `update_item_notes` with `notes`.

4b. CASUAL QUANTITY ADJUSTMENTS (MAKE IT 2 / ADD ONE MORE):
   - When the customer asks to change quantity or says "make it 2", "make that 3", "change to 2", "make the latte 2":
     --> Call `update_quantity(quantity=...)` (optionally with `product_name`).
   - When the customer casually asks to add more (e.g. "add one more", "add another one", "one more please", "add 2 more"):
     --> Call `add_more(quantity=...)` (optionally with `product_name`).
   - NEVER call `search_product` or `add_item` with product_name="2" or "more"!

5. ADD-ONS (VERY IMPORTANT):
   - After calling add_item and the result contains "addon_eligible": true, you MUST immediately call `get_addons()`.
   - Then ask the customer ONE short question: "Would you like to add Non-Dairy Milk, Extra Fruit, Extra Sauce, Extra Soft, or Extra Shot for 0.5 BHD each?"
   - When customer agrees to any add-on, call `add_addon` with that add-on name.
   - If customer says no, move on. NEVER repeat the add-on question again for that item.
   - Eligible items: Affogato, Americano Float, Americano Hot L, Americano Hot M, Cafe Latte Float, Cappuccino M, Cappuccino L, Cappuccino Hot M, Cappuccino Hot L, Caramel Banana Soft Ice, Caramel Latte Float, Caramel Latte Hot L, Caramel Latte Hot M.
   - For any other items, add-ons are NOT available.

6. MOOD-BASED RECOMMENDATIONS:
   - When the customer shares how they feel (e.g. "I'm tired", "need something refreshing", "I want something sweet"), call `recommend_by_mood` with `mood`.
   - Then warmly recommend 1 or 2 matching items in natural spoken voice.

7. FINALIZING ORDER:
   - When the customer is done ordering ("that's all", "nothing else", "check please"), read back the items and total, then ask "Shall I place this order?"
   - Only after a clear confirmation or yes, call `confirm_and_submit`.

8. CRITICAL SPOKEN RULES:
   - NEVER speak tool names, code syntax, brackets, or JSON aloud.
   - Reply directly to the customer in warm, natural English sentences.
   - ALL prices are in BHD (Bahraini Dinar)."""
        self.messages = [{"role": "system", "content": system_content}]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_greeting(self) -> str:
        """Standard warm greeting according to flow."""
        self.welcome_given = True
        greeting = f"Hi, welcome to {self.restaurant_name}! What can I get for you today?"
        self.messages.append({"role": "assistant", "content": greeting})
        return greeting

    def limit_to_two_sentences(self, text: str) -> str:
        """Enforces the strict rule: at most 2 sentences per turn."""
        if not text:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            return " ".join(sentences[:2])
        return " ".join(sentences)

    def _quick_reply(self, text: str, user_input: str) -> str:
        """Record a fast pre-LLM reply into message history and return it."""
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": text})
        return text

    # ------------------------------------------------------------------
    # Pre-LLM guardrails
    # ------------------------------------------------------------------

    def _handle_call_staff(self, cleaned_input: str) -> Optional[str]:
        """If customer explicitly asks for a human, dispatch immediately."""
        if _CALL_STAFF_RE.search(cleaned_input):
            self.dispatcher.dispatch("call_staff", {"reason": "Customer requested human staff"})
            reply = "Of course! I've let the team know — someone will be with you shortly."
            return self._quick_reply(reply, cleaned_input)
        return None

    def _handle_identity(self, cleaned_input: str) -> Optional[str]:
        """Warmly introduce Chef and redirect to ordering."""
        if _IDENTITY_RE.search(cleaned_input) and not _RESTAURANT_RE.search(cleaned_input):
            template = random.choice(_IDENTITY_REPLIES)
            reply = template.format(name=self.restaurant_name)
            return self._quick_reply(reply, cleaned_input)
        return None

    def _handle_off_topic(self, cleaned_input: str) -> Optional[str]:
        """
        Detect clearly off-topic input and give a warm redirect.
        Escalates to touch-screen / call-staff after 2 consecutive off-topic turns.
        Restaurant-related signals override the off-topic detection.
        """
        if not _OFF_TOPIC_RE.search(cleaned_input):
            return None
        if _RESTAURANT_RE.search(cleaned_input):
            return None  # Contains restaurant keywords — let the LLM handle it

        self.off_topic_count += 1
        if self.off_topic_count >= 2:
            self.off_topic_count = 0
            self.dispatcher.dispatch("call_staff", {"reason": "Customer repeatedly off-topic"})
            return self._quick_reply(_OFF_TOPIC_ESCALATE, cleaned_input)

        reply = random.choice(_OFF_TOPIC_REPLIES)
        return self._quick_reply(reply, cleaned_input)

    def _handle_clear_cart(self, cleaned_input: str) -> Optional[str]:
        """Handles requests to clear the entire cart or remove everything."""
        if _CLEAR_CART_RE.search(cleaned_input):
            if self.cart.is_empty():
                reply = "Your cart is already empty! What would you like to order today?"
            else:
                self.cart.clear()
                reply = "I've removed everything from your cart. Your order tray is now empty. What can I get for you?"
            return self._quick_reply(reply, cleaned_input)
        return None

    def _handle_remove_item(self, cleaned_input: str) -> Optional[str]:
        """Handles requests to remove a specific item from the cart, respecting quantity or complete removal."""
        rem_m = _REMOVE_ITEM_RE.match(cleaned_input)
        if rem_m:
            candidate = rem_m.group(1).strip()
            if candidate.lower() in {
                "everything", "all", "all items", "all the items", "cart", "the cart", "my cart",
                "everything from cart", "everything from my cart", "all items from cart"
            }:
                return self._handle_clear_cart("clear cart")

            # Check for complete removal keyword
            completely = False
            if re.search(r'\b(?:completely|all|all\s+of\s+them|all\s+of\s+it|entirely|whole)\b', candidate, re.IGNORECASE):
                completely = True
                candidate = re.sub(r'\b(?:completely|all|all\s+of\s+them|all\s+of\s+it|entirely|whole)\b', '', candidate, flags=re.IGNORECASE).strip()

            WORD_TO_NUM = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            clean_cand = re.sub(r'^(?:the|my|a|an)\s+', '', candidate, flags=re.IGNORECASE).strip()
            q_m = re.match(r'^(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:of\s+)?(?:the\s+)?(.+)$', clean_cand, re.IGNORECASE)
            quantity = None if completely else 1
            if q_m:
                q_str = q_m.group(1).lower()
                parsed_qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM.get(q_str, 1)
                clean_cand = q_m.group(2).strip()
                if not completely:
                    quantity = parsed_qty

            res = self.cart.remove_item(item_identifier=clean_cand, quantity=quantity)
            if res.get("success"):
                item_name = res.get("removed_item", clean_cand)
                qty_removed = res.get("quantity_removed", 1)
                is_completely = res.get("removed_completely", False)
                if self.cart.is_empty():
                    reply = f"I've removed {item_name} from your cart. Your order tray is now empty. What can I get for you?"
                elif is_completely:
                    reply = f"I've removed {item_name} from your cart. You still have: {self.cart.get_items_summary()}. Would you like anything else?"
                else:
                    reply = f"I've removed {qty_removed} {item_name} from your cart. You still have: {self.cart.get_items_summary()}. Would you like anything else?"
            else:
                reply = f"I couldn't find {clean_cand} in your cart. You currently have: {self.cart.get_items_summary()}."
            return self._quick_reply(reply, cleaned_input)
        return None

    def _handle_options_query(self, cleaned_input: str) -> Optional[str]:
        """
        Immediately responds with a warm greeting and lists category items with BHD prices.
        Guarantees zero customer waiting time and prevents the LLM from outputting a bare 'Sure'.
        """
        if not _OPTIONS_QUERY_RE.search(cleaned_input):
            return None

        # Check if customer asked for a specific category
        matched_cat = None
        for cat_name, c_pat in _CATEGORY_MAP:
            if c_pat.search(cleaned_input):
                matched_cat = cat_name
                break

        if matched_cat:
            res = self.menu_client.get_products(category_name=matched_cat)
            prods = res.get("products", [])
            if not prods:
                from menu_api import SAMPLE_PRODUCTS
                prods = [p for p in SAMPLE_PRODUCTS if p.get("category", "").upper() == matched_cat.upper() and p.get("isActive", True)]

            if prods:
                top = prods[:3]
                parts = []
                for p in top:
                    name = p.get("name", "")
                    price = float(p.get("price", 0.0))
                    price_str = f"{price:.1f}" if price == round(price, 1) else f"{price:.3f}".rstrip('0').rstrip('.')
                    parts.append(f"{name} for {price_str} BHD")
                if len(parts) >= 2:
                    items_speech = ", ".join(parts[:-1]) + f", and {parts[-1]}"
                else:
                    items_speech = parts[0]

                has_more = len(prods) > 3
                more_suffix = ", plus more options on the screen" if has_more else ""

                if matched_cat == "HOT DRINKS":
                    reply = f"We have wonderful hot drinks! Some popular choices are {items_speech}{more_suffix}. Which one would you like?"
                elif matched_cat == "COLD DRINKS":
                    reply = f"We have refreshing cold drinks! Popular choices include {items_speech}{more_suffix}. Which one would you like to try?"
                elif matched_cat == "SOFT ICE":
                    reply = f"Here are delicious soft ice options: {items_speech}{more_suffix}. Which flavor would you like?"
                elif matched_cat == "FLOAT":
                    reply = f"Our ice cream floats include {items_speech}{more_suffix}. Which one sounds good to you?"
                elif matched_cat == "SHAKES & AFFOGATO":
                    reply = f"Our shakes and affogato include {items_speech}{more_suffix}. Would you like to try one?"
                elif matched_cat == "KAKIGORI":
                    reply = f"We offer refreshing kakigori shaved ice like {items_speech}{more_suffix}. Which one would you like?"
                else:
                    reply = f"Here are great options in our {matched_cat.lower()}: {items_speech}{more_suffix}. What would you like?"
                return self._quick_reply(reply, cleaned_input)

        # If customer asked generally for drinks
        if re.search(r'\bdrinks?\b|\bbeverages?\b', cleaned_input, re.IGNORECASE):
            reply = "We offer both cold and hot drinks, like Ice Americano for 1.500 BHD, Americano Hot M for 1.500 BHD, and Cappuccino Hot L for 2.400 BHD. Would you like something hot or iced?"
            return self._quick_reply(reply, cleaned_input)

        # General menu request
        reply = "We offer Cold Drinks, Hot Drinks, Soft Ice, Floats, Shakes, and Kakigori! Which category would you like to explore?"
        return self._quick_reply(reply, cleaned_input)

    def _handle_checkout_and_confirmation(self, cleaned_input: str) -> Optional[str]:
        """
        Handles order finalization and customer confirmation:
        1. Confirmation of pending item addition ("I found X for Y BHD. Shall I add that to your cart?").
        2. Final checkout and order placement confirmation ("Shall I place this order?").
        """
        # Step 0: Customer responds to pending item addition confirmation
        if getattr(self, "pending_item_to_add", None):
            if _CONFIRMATION_YES_RE.match(cleaned_input) or re.search(r'\b(?:add|yes|sure|please|confirm|ok|okay|yeah|yep)\b', cleaned_input, re.IGNORECASE):
                pending = self.pending_item_to_add
                self.pending_item_to_add = None
                self.cart.add_item(
                    product_id=pending.get("product_id", ""),
                    name=pending["name"],
                    quantity=pending.get("quantity", 1),
                    price=pending.get("price", 0.0),
                    notes=pending.get("notes", "")
                )
                reply = f"I've added {pending['name']} to your cart. Would you like anything else?"
                return self._quick_reply(reply, cleaned_input)

            elif _CONFIRMATION_NO_RE.match(cleaned_input) or re.search(r'\b(?:no|don\x27t|cancel|stop|nope)\b', cleaned_input, re.IGNORECASE):
                self.pending_item_to_add = None
                reply = "No problem, I won't add that. What else can I get for you?"
                return self._quick_reply(reply, cleaned_input)

        # Step 2: Customer responds to confirmation question
        if getattr(self.cart, "confirmation_requested", False):
            is_confirm = (
                _CONFIRMATION_YES_RE.match(cleaned_input)
                or _PLACE_ORDER_RE.search(cleaned_input)
                or bool(re.search(r'\b(?:yes|place|confirm|submit|proceed|sure|okay|ok|go ahead|do it)\b', cleaned_input, re.IGNORECASE))
            )
            if is_confirm:
                res = self.cart.confirm_and_submit()
                if res.get("success"):
                    order_num = res.get("order_number")
                    total = res.get("total")
                    reply = f"Thank you! Your order #{order_num} has been placed for {total:.3f} BHD. Please proceed to the counter."
                else:
                    reply = res.get("error", "Your cart is currently empty! What would you like to order?")
                return self._quick_reply(reply, cleaned_input)

            elif _CONFIRMATION_NO_RE.match(cleaned_input) or bool(re.search(r'\b(?:no|wait|cancel|stop|not yet|hold on)\b', cleaned_input, re.IGNORECASE)):
                self.cart.confirmation_requested = False
                reply = "No problem! What would you like to add or change in your order?"
                return self._quick_reply(reply, cleaned_input)

        # Step 1: Customer asks to place order / checkout
        if _PLACE_ORDER_RE.search(cleaned_input):
            if self.cart.is_empty():
                reply = "Your order tray is currently empty! What would you like to order today?"
                return self._quick_reply(reply, cleaned_input)

            items_speech = self.cart.get_spoken_items_summary()
            total = self.cart.get_total()
            self.cart.confirmation_requested = True
            reply = f"You have {items_speech}, for a total of {total:.3f} BHD. Shall I place this order?"
            return self._quick_reply(reply, cleaned_input)

        return None

    def _handle_add_item_intent(self, cleaned_input: str) -> Optional[str]:
        """
        Handles explicit ordering / add-item requests.
        If the matched product is fuzzy or detected (e.g. 'choclate shake' -> 'Chocolate Milk Shakes'),
        asks customer for confirmation before adding to cart!
        If exact match, adds directly to cart.
        """
        add_m = _ADD_ITEM_INTENT_RE.match(cleaned_input)
        if not add_m:
            return None

        # Exclude browsing, clear, and remove requests
        if _OPTIONS_QUERY_RE.search(cleaned_input) or _CLEAR_CART_RE.search(cleaned_input) or _REMOVE_ITEM_RE.match(cleaned_input):
            return None

        raw_candidate = add_m.group(1).strip()
        if raw_candidate.lower() in {"this", "that", "it", "something", "anything", "food", "drink", "options"}:
            return None

        WORD_TO_NUM = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        clean_target = re.sub(r'^(?:the|my|a|an)\s+', '', raw_candidate, flags=re.IGNORECASE).strip()
        q_m = re.match(r'^(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:of\s+)?(?:the\s+)?(.+)$', clean_target, re.IGNORECASE)
        qty = 1
        if q_m:
            q_str = q_m.group(1).lower()
            qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM.get(q_str, 1)
            clean_target = q_m.group(2).strip()

        # Search menu catalog for best matching product
        s_res = self.menu_client.search_product(clean_target)
        if not s_res.get("found") or not s_res.get("products"):
            return None

        products = s_res["products"]
        top = products[0]
        top_name = top["name"]
        top_price = float(top.get("price", 0.0))
        top_id = top.get("productId", "")

        # Robust exact-match check using word overlap ratio:
        # The customer's words must cover at least 80% of the product's words,
        # AND the product name (lowercased) must be an exact or near-exact substring of the query.
        q_words = set(re.findall(r'\w+', clean_target.lower()))
        p_words = set(re.findall(r'\w+', top_name.lower()))
        stopwords = {"a", "an", "the", "of", "with", "and"}
        q_words -= stopwords
        p_words -= stopwords

        # Word overlap: what fraction of product name words appear in the customer query
        overlap = len(p_words & q_words) / max(len(p_words), 1)
        # Exact name containment
        name_exact = (clean_target.lower() == top_name.lower() or top_name.lower() in clean_target.lower())
        # Score gap: top result must be clearly dominant over runner-up (if any)
        score_dominant = True
        if len(products) > 1:
            # We don't have scores here, but we can use the is_exact_match flag from search results
            score_dominant = s_res.get("is_exact_match", False) or overlap >= 0.9

        is_exact = name_exact or (overlap >= 0.80 and score_dominant)

        if is_exact:
            self.cart.add_item(product_id=top_id, name=top_name, quantity=qty, price=top_price)
            reply = f"I've added {top_name} to your order. Would you like anything else?"
            return self._quick_reply(reply, cleaned_input)
        else:
            # Fuzzy / detected match: Ask customer confirmation before adding to cart!
            self.pending_item_to_add = {
                "product_id": top_id,
                "name": top_name,
                "quantity": qty,
                "price": top_price,
                "notes": ""
            }
            qty_phrase = f"{qty} " if qty > 1 else ""
            reply = f"I found {top_name} for {top_price:.3f} BHD. Shall I add {qty_phrase}that to your cart?"
            return self._quick_reply(reply, cleaned_input)

        return None

    def _handle_quantity_adjustment(self, cleaned_input: str) -> Optional[str]:
        """
        Handles casual quantity adjustments like:
        - 'make it 2', 'make that 3', 'make it two', 'change to 2', 'make the latte 2'
        - 'add one more', 'one more', 'add another one', 'give me one more', 'add 2 more', 'another one please'
        """
        WORD_TO_NUM = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

        t = cleaned_input.strip().lower()
        t = re.sub(r'\bplease\b', '', t).strip()
        t = re.sub(r'^(?:can\s+(?:i|you)\s+(?:have|get|make|change|add)?|could\s+(?:i|you)\s+(?:have|get|make|change|add)?|actually\s+)', '', t).strip()

        # Check 1: Setting absolute quantity (make it 2, change to 3, etc.)
        target_qty = None
        target_item = None

        m_make = re.match(r'^make\s+(?:it|that|this)(?:\s+into)?\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+(?:of\s+them|of\s+those|instead))?$', t)
        if m_make:
            q_str = m_make.group(1)
            target_qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM.get(q_str, 1)

        if target_qty is None:
            m_change = re.match(r'^change\s+(?:(?:it|that|this)\s+)?(?:the\s+)?(?:quantity\s+)?to\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+instead)?$', t)
            if m_change:
                q_str = m_change.group(1)
                target_qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM.get(q_str, 1)

        if target_qty is None:
            m_named = re.match(r'^make\s+(?:the\s+)?(.+?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)$', t)
            if m_named:
                cand_item = m_named.group(1).strip()
                if cand_item not in {"it", "that", "this"}:
                    q_str = m_named.group(2)
                    target_qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM.get(q_str, 1)
                    target_item = cand_item

        if target_qty is not None:
            if self.cart.is_empty():
                reply = "Your order tray is currently empty! What would you like to order today?"
                return self._quick_reply(reply, cleaned_input)

            res = self.cart.set_item_quantity(quantity=target_qty, item_identifier=target_item)
            if res.get("success"):
                p_name = res.get("product_name", "item")
                if res.get("action") == "removed":
                    if self.cart.is_empty():
                        reply = f"I've removed {p_name} from your cart. Your order tray is now empty. What can I get for you?"
                    else:
                        reply = f"I've removed {p_name} from your cart. You still have: {self.cart.get_items_summary()}. Would you like anything else?"
                else:
                    reply = f"I've updated {p_name} to {target_qty}. Total is {self.cart.get_total():.3f} BHD. Would you like anything else?"
                return self._quick_reply(reply, cleaned_input)
            else:
                reply = f"I couldn't find that item in your cart. You currently have: {self.cart.get_items_summary()}."
                return self._quick_reply(reply, cleaned_input)

        # Check 2: Relative quantity increase (add one more, another one, add 2 more, etc.)
        delta_qty = None
        delta_item = None

        m_add_one = re.match(r'^(?:add|give\s+me|get\s+me|have)?\s*(?:an?other\s+one|an?other|one\s+more|1\s+more)(?:\s+(?:of\s+)?(?:the\s+)?(.+?))?$', t)
        if m_add_one:
            delta_qty = 1
            cand = m_add_one.group(1)
            if cand and cand not in {"please", "it", "that", "this", "them", "those"}:
                delta_item = cand

        if delta_qty is None:
            m_add_n = re.match(r'^(?:add|give\s+me|get\s+me|have)?\s*(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+more(?:\s+(?:of\s+)?(?:the\s+)?(.+?))?$', t)
            if m_add_n:
                q_str = m_add_n.group(1)
                delta_qty = int(q_str) if q_str.isdigit() else WORD_TO_NUM.get(q_str, 1)
                cand = m_add_n.group(2)
                if cand and cand not in {"please", "it", "that", "this", "them", "those"}:
                    delta_item = cand

        if delta_qty is not None:
            if self.cart.is_empty():
                reply = "Your order tray is currently empty! What would you like to order today?"
                return self._quick_reply(reply, cleaned_input)

            res = self.cart.adjust_item_quantity(delta=delta_qty, item_identifier=delta_item)
            if res.get("success"):
                p_name = res.get("product_name", "item")
                new_q = res.get("new_quantity", 1)
                if delta_qty == 1:
                    reply = f"I've added another {p_name}. You now have {new_q} in your cart for {self.cart.get_total():.3f} BHD. Would you like anything else?"
                else:
                    reply = f"I've added {delta_qty} more {p_name}. You now have {new_q} in your cart for {self.cart.get_total():.3f} BHD. Would you like anything else?"
                return self._quick_reply(reply, cleaned_input)
            else:
                reply = f"I couldn't find that item in your cart. You currently have: {self.cart.get_items_summary()}."
                return self._quick_reply(reply, cleaned_input)

        return None

    # ------------------------------------------------------------------
    # Main turn processing
    # ------------------------------------------------------------------

    def process_turn(self, customer_utterance: str) -> str:
        """
        Process one customer turn and return Chef's spoken response.
        Fast-path checks run before the LLM to minimize latency.
        """
        cleaned_input = customer_utterance.strip()

        # ── 1. Silent / empty / repetition hallucination ──────────────────
        if not cleaned_input or is_repetition_hallucination(cleaned_input):
            self.misunderstand_count += 1
            if self.misunderstand_count >= 2:
                self.dispatcher.dispatch("call_staff", {"reason": "Customer audio unclear / repetitive noise"})
                return "I'm having a little trouble hearing you — a staff member is on their way to help."
            return "I didn't quite catch that. What would you like to order?"

        # ── 2. Simple greetings & common messages (no LLM needed) ───────
        simple = cleaned_input.lower().strip('.!? ')
        greeting_words = {"hi", "hello", "hey", "good morning", "good afternoon",
                          "good evening", "hi there", "hello there", "salam", "marhaba",
                          "hi chef", "hello chef", "hey chef"}
        if simple in greeting_words or re.match(r'^(?:hi|hello|hey)\b(?:\s+(?:there|chef|friend))?$', simple):
            if self.welcome_given:
                reply = "Hello! What can I get started for you today?"
            else:
                self.welcome_given = True
                reply = f"Hi there! Welcome to {self.restaurant_name}. What can I get for you today?"
            self.off_topic_count = 0
            self.misunderstand_count = 0
            return self._quick_reply(reply, cleaned_input)

        # ── 3. Call-staff shortcut ────────────────────────────────────────
        result = self._handle_call_staff(cleaned_input)
        if result is not None:
            self.misunderstand_count = 0
            return result

        # ── 3a. Clear cart / remove everything shortcut ──────────────────
        clear_res = self._handle_clear_cart(cleaned_input)
        if clear_res is not None:
            self.misunderstand_count = 0
            self.off_topic_count = 0
            return clear_res

        # ── 3b. Remove specific item shortcut ─────────────────────────────
        rem_res = self._handle_remove_item(cleaned_input)
        if rem_res is not None:
            self.misunderstand_count = 0
            self.off_topic_count = 0
            return rem_res

        # ── 3c. Category options & menu browsing shortcut ────────────────
        opt_res = self._handle_options_query(cleaned_input)
        if opt_res is not None:
            self.misunderstand_count = 0
            self.off_topic_count = 0
            return opt_res

        # ── 3d. Checkout & Confirmation shortcut ─────────────────────────
        checkout_res = self._handle_checkout_and_confirmation(cleaned_input)
        if checkout_res is not None:
            self.misunderstand_count = 0
            self.off_topic_count = 0
            return checkout_res

        # ── 3dc. Casual quantity adjustment shortcut (e.g. "make it 2", "add one more") ──
        qty_res = self._handle_quantity_adjustment(cleaned_input)
        if qty_res is not None:
            self.misunderstand_count = 0
            self.off_topic_count = 0
            return qty_res

        # ── 3dd. Add item intent with confirmation for detected items ───
        add_res = self._handle_add_item_intent(cleaned_input)
        if add_res is not None:
            self.misunderstand_count = 0
            self.off_topic_count = 0
            return add_res

        # ── 3e. Customization shortcut (e.g. "less sugar", "no sugar", "extra ice") ──
        if not self.cart.is_empty() and _CUSTOMIZATION_RE.search(cleaned_input):
            self.cart.update_item_notes(cleaned_input)
            last_item = self.cart.items[-1]
            reply = f"I've updated your {last_item.name} with {cleaned_input}. What else can I get for you?"
            self.off_topic_count = 0
            self.misunderstand_count = 0
            return self._quick_reply(reply, cleaned_input)

        # ── 3bb. Add-on shortcut (e.g. "extra shot", "yes with extra shot", "non dairy milk", "extra fruit") ──
        lower_input = cleaned_input.lower()
        addon_patterns = [
            (r'\b(?:non[- ]?dairy\s+milk)\b', "Non-Dairy Milk"),
            (r'\b(?:extra\s+fruit)\b', "Extra Fruit"),
            (r'\b(?:extra\s+sauce)\b', "Extra Sauce"),
            (r'\b(?:extra\s+soft)\b', "Extra Soft"),
            (r'\b(?:extra\s+shot)\b', "Extra Shot"),
        ]
        detected_addon_name = None
        for pat, a_name in addon_patterns:
            if re.search(pat, lower_input):
                detected_addon_name = a_name
                break

        if detected_addon_name and not self.cart.is_empty():
            last_item = self.cart.items[-1]
            if self.menu_client.is_addon_eligible(last_item.name):
                self.cart.add_addon(detected_addon_name, item_identifier=last_item.name, price=0.500)
                reply = f"I've added {detected_addon_name} to your {last_item.name}. What else can I get for you?"
            else:
                reply = f"Sorry, add-ons are only available for select items, not for {last_item.name}. What else can I get for you?"
            self.off_topic_count = 0
            self.misunderstand_count = 0
            return self._quick_reply(reply, cleaned_input)

        # ── 3c. Mood recommendation shortcut (e.g. "tired, what do you recommend?") ──
        lower_input = cleaned_input.lower()
        is_rec_question = any(w in lower_input for w in ["recommend", "suggest", "what should i", "what do you suggest", "what can you suggest", "what is good", "what do you recommend", "what to order", "what to drink", "what to eat"])
        matched_mood = None
        for kw, mood_val in [
            ("tired", "tired"), ("exhaust", "tired"), ("sleepy", "tired"), ("energy", "tired"),
            ("stress", "stress"), ("relax", "stress"), ("calm", "stress"), ("unwind", "stress"),
            ("hot", "hot"), ("thirsty", "refreshing"), ("refresh", "refreshing"),
            ("sweet", "sweet"), ("dessert", "sweet"), ("treat", "sweet"),
            ("health", "healthy"), ("diet", "healthy"), ("light", "healthy")
        ]:
            if kw in lower_input:
                matched_mood = mood_val
                break

        if matched_mood and is_rec_question:
            res_str = self.dispatcher.dispatch("recommend_by_mood", {"mood": matched_mood})
            res = json.loads(res_str)
            recs = res.get("recommendations", [])
            if len(recs) >= 2:
                pick1 = recs[0]["name"]
                pick2 = recs[1]["name"]
                headline = res.get("headline", "Here are great options for you")
                reply = f"{headline}: I recommend our {pick1} or {pick2}. Would you like to try one?"
            elif recs:
                reply = f"I'd recommend our {recs[0]['name']}. Would you like to add that?"
            else:
                reply = "I recommend our Hot Latte or Fresh Apple Juice. What sounds good to you?"
            self.off_topic_count = 0
            self.misunderstand_count = 0
            return self._quick_reply(reply, cleaned_input)

        # ── 4. Identity question shortcut ─────────────────────────────────
        result = self._handle_identity(cleaned_input)
        if result is not None:
            self.off_topic_count = 0
            self.misunderstand_count = 0
            return result

        # ── 5. Off-topic detection ────────────────────────────────────────
        result = self._handle_off_topic(cleaned_input)
        if result is not None:
            self.misunderstand_count = 0
            return result

        # ── 6. Restaurant ordering — full LLM tool-calling loop ───────────
        # Reset counters on a valid restaurant turn
        self.off_topic_count = 0

        # Track declined suggestions
        lower_input = cleaned_input.lower()
        if self.last_offered_suggestion and any(
            neg in lower_input for neg in ["no", "nope", "don't want", "nah", "skip", "not that"]
        ):
            self.cart.record_declined_suggestion(self.last_offered_suggestion)
            self.last_offered_suggestion = None

        self.messages.append({"role": "user", "content": cleaned_input})

        max_tool_iterations = 6
        iteration = 0
        final_reply = ""

        while iteration < max_tool_iterations:
            iteration += 1

            # Sliding window: system prompt + last 20 messages (user-chosen window size)
            system_msg = self.messages[:1]
            recent_msgs = self.messages[1:]
            if len(recent_msgs) > 20:
                recent_msgs = recent_msgs[-20:]
            windowed_messages = system_msg + recent_msgs

            chat_url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": windowed_messages,
                "tools": TOOL_DEFINITIONS,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 256,   # Cap token output — Chef responses are short
                }
            }

            try:
                resp = requests.post(chat_url, json=payload, timeout=30)
                if resp.status_code != 200:
                    logger.error(f"Ollama API error {resp.status_code}: {resp.text}")
                    return (
                        "I'm having a little trouble right now. "
                        "Please use the touch screen or I can call a staff member for you."
                    )

                res_data = resp.json()
                msg = res_data.get("message", {})
                tool_calls = msg.get("tool_calls")
                content = msg.get("content", "")

                # Resilient fallback: If model emitted tool call into text content
                if not tool_calls and content:
                    tool_calls = _extract_tool_calls_from_text(content)

                # Model requested tool calls
                if tool_calls:
                    # Keep content empty in message history so model does not parrot code in subsequent turns
                    self.messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    })

                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        fn_args = fn.get("arguments", {})
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except Exception:
                                fn_args = {}

                        logger.info(f"Chef executing tool: {fn_name}({fn_args})")
                        tool_result = self.dispatcher.dispatch(fn_name, fn_args)

                        if "API error" in tool_result or "Unauthorized" in tool_result:
                            logger.warning(f"Tool {fn_name} reported API issue.")

                        self.messages.append({
                            "role": "tool",
                            "name": fn_name,
                            "content": tool_result,
                        })
                    continue  # Feed results back into model

                else:
                    # Model produced a final text response
                    final_reply = content
                    break

            except requests.exceptions.Timeout:
                logger.error(f"Ollama request to {chat_url} timed out.")
                return (
                    "Sorry, I'm taking too long to respond. "
                    "Please use the touch screen or I'll get a staff member."
                )
            except Exception as e:
                logger.error(f"Error communicating with Ollama at {chat_url}: {e}")
                return "I ran into a problem. Would you like to use the touch screen or speak with staff?"

        if not final_reply:
            final_reply = "I'm here to help with your order — what can I get for you?"

        # Anti-stall guardrail: If customer asked for options/menu but LLM gave conversational filler like "Sure" without items
        if _OPTIONS_QUERY_RE.search(cleaned_input):
            lower_r = final_reply.lower()
            words = lower_r.split()
            if len(words) <= 12 and any(w in lower_r for w in ["sure", "certainly", "of course", "yes", "what would you like", "what can i get", "what sounds good"]):
                if not any(kw in lower_r for kw in ["bhd", "price", "dinar"]):
                    opt_res = self._handle_options_query(cleaned_input)
                    if opt_res:
                        return opt_res

        # Removal / clearing synchronization guardrail
        lower_r = final_reply.lower()
        if any(w in lower_r for w in ["removed", "removing", "taken off", "cleared", "emptied", "cancelling", "cancelled"]):
            if any(w in cleaned_input.lower() for w in ["everything", "all", "cart"]):
                self.cart.clear()
            else:
                rem_m = _REMOVE_ITEM_RE.match(cleaned_input)
                if rem_m:
                    cand = rem_m.group(1).strip()
                    self.cart.remove_item(cand)

        # ── Post-process: strip markdown → transform leaked code into dialogue → condense runaway lists → limit to 2 sentences ──
        clean_text = clean_spoken_text(final_reply)
        friendly_text = format_friendly_reply_if_code(clean_text, self.cart)
        condensed_text = condense_long_spoken_list(friendly_text)
        voice_text = self.limit_to_two_sentences(condensed_text)

        # Append final response to history
        self.messages.append({"role": "assistant", "content": voice_text})
        self.misunderstand_count = 0

        return voice_text
