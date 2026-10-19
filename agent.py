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
    "call_staff": "call_staff",
    "callstaff": "call_staff",
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
        self.last_offered_suggestion: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []

        self._init_system_prompt()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _init_system_prompt(self):
        system_content = f"""You are "Chef", the friendly voice ordering assistant at {self.restaurant_name} restaurant kiosk.
Everything you say is spoken aloud through a speaker, so NEVER use lists, bullet points, numbers (like 1. or 2.), symbols, or markdown.
Keep all spoken replies warm, natural, and AT MOST 2 SHORT SENTENCES.

MANDATORY TOOL EXECUTION RULES (CRITICAL):
1. ORDERING / ADDING ITEMS:
   - When the customer asks to order, eat, or drink anything (e.g. "I want green tea", "add hot latte", "apple juice", "green tea", "two coffees", "a burger"), you MUST call the `add_item` tool with `product_name`, `quantity`, and any `notes` (such as "less sugar", "extra ice").
   - NEVER call search_product when the customer is ordering or asking for an item (like "I want green tea")—call `add_item` directly!
   - NEVER say or claim you added an item in conversational text without calling `add_item` first! The physical cart on the screen will not update unless you emit the tool call!
   - You do NOT need to call search_product first—`add_item` automatically matches and prices the item for you.
2. REMOVING / CANCELING ITEMS:
   - When the customer asks to remove, cancel, or take off an item (e.g. "remove the hot latte", "take off the coffee", "cancel one latte", "remove one item from cart"), you MUST call the `remove_item` tool with `product_name` and `quantity`.
   - NEVER claim you removed an item without calling `remove_item` first!
3. CUSTOMIZING ITEMS (SWEETNESS / ICE / NOTES):
   - When the customer specifies sweetness or customization for an item (e.g. "less sugar", "no sugar", "extra ice", "oat milk"), call `update_item_notes` with `notes`.
4. MOOD-BASED RECOMMENDATIONS (IMPORTANT BUSINESS STRATEGY):
   - When the customer shares how they feel, their mood, or an occasion (e.g. "I'm tired", "need an energy boost", "I'm stressed", "hot outside", "need something refreshing", "I want something sweet", "healthy choice", "recommend for my mood"), you MUST call `recommend_by_mood` with `mood`.
   - Then warmly recommend 1 or 2 matching items in natural spoken voice.
5. BROWSING / ASKING FOR OPTIONS:
   - When the customer asks what you have or asks for a category (e.g. "what drinks do you have?", "what is on the menu?"), call `get_categories`, `get_products`, or `search_product`. Mention at most 3 options briefly.
6. FINALIZING ORDER:
   - When the customer is done ordering ("that's all", "nothing else", "check please"), read back the items and total, then ask "Shall I place this order?"
   - Only after a clear confirmation or yes, call `confirm_and_submit`.
7. CRITICAL SPOKEN RULES:
   - NEVER speak tool names, code syntax (such as add_item, additem, {{"notes": ...}}), brackets, or JSON aloud.
   - Reply directly to the customer in warm, natural English sentences."""
        self.messages = [{"role": "system", "content": system_content}]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_greeting(self) -> str:
        """Standard warm greeting according to flow."""
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

        # ── 2. Simple greetings (no LLM needed) ──────────────────────────
        simple = cleaned_input.lower().strip('.!? ')
        if simple in {"hi", "hello", "hey", "good morning", "good afternoon",
                      "good evening", "hi there", "hello there", "salam", "marhaba"}:
            reply = f"Hi there! Welcome to {self.restaurant_name}. What can I get for you today?"
            self.off_topic_count = 0
            self.misunderstand_count = 0
            return self._quick_reply(reply, cleaned_input)

        # ── 3. Call-staff shortcut ────────────────────────────────────────
        result = self._handle_call_staff(cleaned_input)
        if result is not None:
            self.misunderstand_count = 0
            return result

        # ── 3b. Customization shortcut (e.g. "less sugar", "no sugar", "extra ice") ──
        if not self.cart.is_empty() and _CUSTOMIZATION_RE.search(cleaned_input):
            self.cart.update_item_notes(cleaned_input)
            last_item = self.cart.items[-1]
            reply = f"I've updated your {last_item.name} with {cleaned_input}. What else can I get for you?"
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
                logger.error("Ollama request timed out.")
                return (
                    "Sorry, I'm taking too long to respond. "
                    "Please use the touch screen or I'll get a staff member."
                )
            except Exception as e:
                logger.error(f"Error communicating with Ollama: {e}")
                return "I ran into a problem. Would you like to use the touch screen or speak with staff?"

        if not final_reply:
            final_reply = "I'm here to help with your order — what can I get for you?"

        # ── Post-process: strip markdown → transform leaked code into dialogue → limit to 2 sentences ──
        clean_text = clean_spoken_text(final_reply)
        friendly_text = format_friendly_reply_if_code(clean_text, self.cart)
        voice_text = self.limit_to_two_sentences(friendly_text)

        # Append final response to history
        self.messages.append({"role": "assistant", "content": voice_text})
        self.misunderstand_count = 0

        return voice_text
