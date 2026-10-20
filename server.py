import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import soundfile as sf
import numpy as np

# Ensure workspace is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from cart import OrderCart
from menu_api import MenuAPIClient
from agent import ChefAgent
from vad import SileroVADDetector
from stt import WhisperSTT
from tts import TextToSpeech

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("KioskServer")

app = FastAPI(title="Girfalco Kiosk Voice Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
cart = OrderCart()
menu_client = MenuAPIClient(force_mock=False)
agent = ChefAgent(cart=cart, menu_client=menu_client)
tts = TextToSpeech()

# Lazy loaded audio detectors
vad_detector: Optional[SileroVADDetector] = None
whisper_stt: Optional[WhisperSTT] = None

def get_vad() -> SileroVADDetector:
    global vad_detector
    if vad_detector is None:
        logger.info("Initializing Silero VAD v5 detector...")
        vad_detector = SileroVADDetector()
    return vad_detector

def get_stt() -> WhisperSTT:
    global whisper_stt
    if whisper_stt is None:
        logger.info("Initializing Whisper STT...")
        whisper_stt = WhisperSTT()
    return whisper_stt

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for d in dead:
            if d in self.active_connections:
                self.active_connections.remove(d)

    def broadcast_sync(self, message: Dict[str, Any]):
        """Thread-safe synchronous broadcast call from background audio workers."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)

ws_manager = ConnectionManager()

# Background Voice Loop Worker State
voice_worker_running = False
voice_worker_thread: Optional[threading.Thread] = None

def _cart_to_dict(c: OrderCart) -> Dict[str, Any]:
    return c.to_dict()

class TextChatRequest(BaseModel):
    message: str
    cart: Optional[Dict[str, Any]] = None

class CartSyncRequest(BaseModel):
    cart: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    ws_manager.set_loop(asyncio.get_event_loop())
    # Deliver initial greeting into agent memory
    greeting = agent.get_greeting()
    logger.info(f"Kiosk Agent initialized. Initial greeting: '{greeting}'")

@app.get("/api/status")
def get_status():
    return {
        "restaurant_name": Config.RESTAURANT_NAME,
        "model": Config.OLLAMA_MODEL,
        "tts_engine": Config.TTS_ENGINE,
        "whisper_model": Config.WHISPER_MODEL,
        "vad_min_silence_ms": Config.VAD_MIN_SILENCE_MS,
        "voice_loop_active": voice_worker_running,
        "cart": _cart_to_dict(cart)
    }

@app.get("/api/cart")
def get_cart_state():
    """Retrieve current in-memory cart state."""
    return _cart_to_dict(cart)

@app.get("/api/menu")
def get_menu():
    """Retrieve category and product catalog from Menu API."""
    cats_res = menu_client.get_categories()
    prods_res = menu_client.get_products()
    return {
        "categories": cats_res.get("categories", []),
        "products": prods_res.get("products", [])
    }

@app.post("/api/chat")
async def send_text_chat(req: TextChatRequest):
    """Simple HTTP fallback for text turns."""
    user_text = req.message.strip()
    if not user_text:
        return {"error": "Empty message"}

    # Load active cart sent from frontend (client-side state)
    if req.cart is not None:
        cart.load_from_dict(req.cart)

    # Broadcast customer message
    await ws_manager.broadcast({
        "type": "chat_message",
        "speaker": "Customer",
        "text": user_text,
        "timestamp": time.strftime("%H:%M:%S")
    })

    # Broadcast thinking state
    await ws_manager.broadcast({
        "type": "vad_state",
        "state": "THINKING",
        "detail": "Chef checking menu and preparing reply..."
    })

    # Process through agent
    reply = agent.process_turn(user_text)

    # Broadcast agent reply
    await ws_manager.broadcast({
        "type": "chat_message",
        "speaker": "Chef",
        "text": reply,
        "timestamp": time.strftime("%H:%M:%S")
    })

    # Broadcast updated cart to all clients
    await ws_manager.broadcast({
        "type": "cart_update",
        "cart": _cart_to_dict(cart)
    })

    # Broadcast idle state
    await ws_manager.broadcast({
        "type": "vad_state",
        "state": "IDLE",
        "detail": "Ready"
    })

    return {
        "reply": reply,
        "cart": _cart_to_dict(cart)
    }

def _run_single_voice_turn():
    vad = get_vad()
    stt = get_stt()

    ws_manager.broadcast_sync({
        "type": "vad_state",
        "state": "LISTENING",
        "hold_ms": Config.VAD_MIN_SILENCE_MS,
        "detail": "Listening for your voice..."
    })

    def on_speech_start():
        ws_manager.broadcast_sync({
            "type": "vad_state",
            "state": "CUSTOMER_SPEAKING",
            "detail": "Speech detected! Recording..."
        })

    def on_speech_end():
        ws_manager.broadcast_sync({
            "type": "vad_state",
            "state": "HOLDING_SILENCE",
            "hold_ms": Config.VAD_MIN_SILENCE_MS,
            "detail": "Natural pause hold..."
        })

    def on_vad_status(status_label: str, speech_prob: float):
        ws_manager.broadcast_sync({
            "type": "vad_level",
            "status": status_label,
            "probability": round(speech_prob, 3)
        })

    audio = vad.listen_and_record_speech(
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
        on_vad_status=on_vad_status,
        timeout=15.0
    )

    if audio is None:
        ws_manager.broadcast_sync({
            "type": "vad_state",
            "state": "IDLE",
            "detail": "No speech detected."
        })
        return

    # Transcribing
    ws_manager.broadcast_sync({
        "type": "vad_state",
        "state": "TRANSCRIBING",
        "detail": "Transcribing speech with Whisper..."
    })

    transcript = stt.transcribe(audio)
    if not transcript:
        ws_manager.broadcast_sync({
            "type": "vad_state",
            "state": "IDLE",
            "detail": "Could not understand audio."
        })
        return

    # Broadcast customer turn
    ws_manager.broadcast_sync({
        "type": "chat_message",
        "speaker": "Customer",
        "text": transcript,
        "timestamp": time.strftime("%H:%M:%S")
    })

    # Thinking
    ws_manager.broadcast_sync({
        "type": "vad_state",
        "state": "THINKING",
        "detail": "Chef thinking..."
    })

    reply = agent.process_turn(transcript)

    # Broadcast Chef turn
    ws_manager.broadcast_sync({
        "type": "chat_message",
        "speaker": "Chef",
        "text": reply,
        "timestamp": time.strftime("%H:%M:%S")
    })

    # Broadcast cart update
    ws_manager.broadcast_sync({
        "type": "cart_update",
        "cart": _cart_to_dict(cart)
    })

    # Speaking aloud
    ws_manager.broadcast_sync({
        "type": "vad_state",
        "state": "CHEF_SPEAKING",
        "detail": "Chef speaking aloud..."
    })

    tts.speak(reply)

    # Settling delay
    time.sleep(0.5)

    ws_manager.broadcast_sync({
        "type": "vad_state",
        "state": "IDLE",
        "detail": "Ready"
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Unified real-time bidirectional WebSocket channel for voice, dialogue turns, and cart updates.
    Eliminates redundant REST POST endpoints by handling events over a single connection.
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial status & cart
        await websocket.send_json({
            "type": "init",
            "restaurant_name": Config.RESTAURANT_NAME,
            "vad_min_silence_ms": Config.VAD_MIN_SILENCE_MS,
            "cart": _cart_to_dict(cart),
            "state": "IDLE"
        })
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                
                # 1. Trigger voice turn on kiosk hardware mic
                if action == "voice_turn":
                    threading.Thread(target=_run_single_voice_turn, daemon=True).start()
                
                # 2. Text turn
                elif action == "text":
                    user_text = msg.get("message", "").strip()
                    if user_text:
                        await ws_manager.broadcast({
                            "type": "chat_message",
                            "speaker": "Customer",
                            "text": user_text,
                            "timestamp": time.strftime("%H:%M:%S")
                        })
                        await ws_manager.broadcast({
                            "type": "vad_state",
                            "state": "THINKING",
                            "detail": "Chef checking menu and preparing reply..."
                        })
                        reply = agent.process_turn(user_text)
                        await ws_manager.broadcast({
                            "type": "chat_message",
                            "speaker": "Chef",
                            "text": reply,
                            "timestamp": time.strftime("%H:%M:%S")
                        })
                        await ws_manager.broadcast({
                            "type": "cart_update",
                            "cart": _cart_to_dict(cart)
                        })
                        await ws_manager.broadcast({
                            "type": "vad_state",
                            "state": "IDLE",
                            "detail": "Ready"
                        })
                
                # 3. Clear cart
                elif action == "clear_cart":
                    cart.clear()
                    await ws_manager.broadcast({
                        "type": "cart_update",
                        "cart": _cart_to_dict(cart)
                    })

                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception as e:
                logger.error(f"Error processing WS event: {e}")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# Serve static frontend files
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
