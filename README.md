# 🍽️ Kiosk Voice Agent — "Chef" Voice Assistant

A complete, production-ready voice ordering assistant built for restaurant self-service kiosks.

"Chef" speaks naturally aloud, understands casual speech and slang ("gimme a burger", "two of those", "no onions"), uses local voice activity detection, calls live REST APIs for menu categories and products, tracks the customer cart, reads back the order, and finalizes submissions.

---

## 🚀 Key Features & Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │           Customer Microphone / Voice        │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        Silero VAD v5 (Local 16kHz Mono)      │
                  │   Detects speech start & end with low latency│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         OpenAI Whisper (Local CUDA/CPU)      │
                  │      Accurate STT preserving casual slang    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Chef Agent (Local Ollama LLM)        │
                  │     llama3.2:3b with function/tool calling   │
                  │  Strict voice rules (<= 2 sentences, no md)  │
                  └──────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌───────────────────────────────┐ ┌───────────────────────────┐
  │     Menu API & Cart Tools     │ │    pyttsx3 Text-To-Speech │
  │ - get_categories              │ │ Sanitizes markdown/symbols│
  │ - get_products (by category)  │ │ Speaks response aloud     │
  │ - search_product (relevance)  │ └───────────────────────────┘
  │ - add_item / get_cart         │
  │ - confirm_and_submit          │
  │ - call_staff                  │
  └───────────────────────────────┘
```

1. **Silero VAD v5 (Local)**: Real-time neural voice activity detection at 16kHz, isolating human speech from kiosk ambient noise with zero cloud dependencies.
2. **OpenAI Whisper (Local CUDA/CPU)**: Transcribes customer speech in milliseconds on GPU.
3. **Chef Conversational Engine**:
   - Spoken style: Warm, natural, concise (maximum 2 sentences per turn).
   - Zero markdown, bullet points, or symbols in speech.
   - Menu data strictly from APIs (never invented).
   - Mentions at most 3 options when general categories are requested.
   - Reads back items and total, requiring explicit confirmation before placing the order.
4. **Resilient API Client**:
   - Connects to `PRODUCTS_CATEGORY_URL` and `PRODUCTS_API` with `API_TOKEN` from `.env.development`.
   - Dynamic token reloading: updates made to `.env.development` apply immediately without restarting.
   - Fallback offline menu for local testing when token is expired or network is unreachable.
5. **Rich Terminal UI**:
   - Visual display of active order cart, quantities, subtotal, and live dialogue.

---

## 📁 File Structure

- [`config.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/config.py): Environment configuration and variable loader.
- [`vad.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/vad.py): Local Silero VAD v5 streaming detector with 512-sample chunks.
- [`stt.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/stt.py): Whisper speech-to-text transcriber on GPU/CPU.
- [`tts.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/tts.py): pyttsx3 speech synthesizer with spoken-text sanitization.
- [`menu_api.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/menu_api.py): REST API client for categories, products, relevance search, and graceful error handling.
- [`cart.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/cart.py): Active order state, items, quantities, subtotal, and declined suggestions tracker.
- [`tools.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/tools.py): Ollama function calling definitions and execution dispatcher.
- [`agent.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/agent.py): Chef agent conversational loop, voice guardrails, and misunderstanding handler.
- [`kiosk_ui.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/kiosk_ui.py): Rich terminal interface for kiosk orders and dialogue.
- [`main.py`](file:///home/force-4/Documents/Kiosk%20Voice%20Agent/main.py): Main application entry point supporting both live microphone voice and interactive text simulation.

---

## ⚙️ Configuration (`.env.development`)

Ensure your `.env.development` file has the following variables:

```env
PRODUCTS_API="http://51.222.143.153:5001/api/v1/products"
PRODUCTS_CATEGORY_URL="http://51.222.143.153:5001/api/v1/product-categories"
API_TOKEN="your_bearer_token_here"
```

Optional settings:
- `OLLAMA_MODEL="llama3.2:3b"`
- `RESTAURANT_NAME="Girfalco"`
- `WHISPER_MODEL="base"`
- `VAD_MIN_SILENCE_MS="800"`

---

## 🎙️ Running the Kiosk Agent

### 1. Live Voice Mode (Microphone + Silero VAD v5 + Whisper + TTS)
Speak directly into the microphone. Silero VAD v5 detects when you speak, Whisper transcribes your voice, Chef processes your order, and speaks back through the kiosk speakers:

```bash
python3 main.py --voice
```

### 2. Interactive Text Simulation Mode
Ideal for testing conversations, slang, menu queries, and tool calls directly in the terminal:

```bash
python3 main.py --text
```
*(Chef's replies will be printed and spoken aloud via TTS. To run silently, add `--no-tts`)*

### 3. Test API Connections
Verify live connection to Categories and Products endpoints:

```bash
python3 main.py --test-api
```

### 4. Test Microphone, Silero VAD v5 & Speaker
Run an audio diagnostics test to verify microphone input, Silero VAD speech detection, Whisper transcription, and TTS audio output:

```bash
python3 main.py --test-audio
```

---

## 📋 Behavioral Compliance Checklist

| Rule / Requirement | Implementation Detail |
|---|---|
| **Spoken Aloud (No markdown/lists)** | `clean_spoken_text()` strips all markdown, bullets, symbols, numbers |
| **Max 2 Sentences** | `limit_to_two_sentences()` enforces <= 2 sentences per turn |
| **Menu from APIs only** | Dynamic query via `get_categories`, `get_products`, `search_product` |
| **Max 3 options for general request** | `search_product` returns top 3 relevance matches and asks which one |
| **Order Confirmation Flow** | Items & total read back; `confirm_and_submit` blocked until clear "yes" |
| **Silero VAD v5** | Loaded locally via PyTorch at 16kHz 512-sample streaming chunks |
| **Call Staff on Issue / Allergies** | `call_staff` tool invoked on unknown allergens, API failures, or 2 silences |
