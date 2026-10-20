import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env.development or .env
ENV_DEV_PATH = BASE_DIR / ".env.development"
ENV_PATH = BASE_DIR / ".env"

if ENV_DEV_PATH.exists():
    load_dotenv(dotenv_path=ENV_DEV_PATH, override=True)
elif ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)


class Config:
    # API endpoints
    PRODUCTS_API = os.getenv("PRODUCTS_API", "http://51.222.143.153:5001/api/v1/products")
    PRODUCTS_CATEGORY_URL = os.getenv("PRODUCTS_CATEGORY_URL", "http://51.222.143.153:5001/api/v1/product-categories")
    API_TOKEN = os.getenv("API_TOKEN", "")
    REFRESH_API_URL = os.getenv("REFRESH_API_URL", "http://51.222.143.153:5001/api/v1/auth/refresh")
    REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")

    # Restaurant Info
    RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "Girfalco")

    # LLM Settings (Ollama)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    # Silero VAD Settings (optimized for natural kiosk conversational pauses without cutting off mid-sentence)
    VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.50"))
    # Minimum silence duration in ms before considering speech finished (1400ms allows natural mid-sentence pauses without cutting off)
    VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "1400"))
    VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "250"))
    # Chunk size for Silero VAD v5 at 16kHz
    VAD_CHUNK_SIZE = 512

    # Whisper STT Settings (small provides high accuracy for accents & customizations, runs in ~0.6s)
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # TTS Settings:
    TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")  # "piper" (primary local neural, ~70ms) or "edge" (cloud neural)

    # 1. Primary Local Neural Voice: Piper TTS (100% offline, ~70ms latency, zero cloud timeouts)
    #    en_US-ryan-medium: Natural American English (Male)
    #    ar_JO-kareem-medium: Arabic (Male)
    PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", str(BASE_DIR / "models" / "piper" / "en_US-ryan-medium.onnx"))
    PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", str(BASE_DIR / "models" / "piper" / "en_US-ryan-medium.onnx.json"))
    PIPER_ARABIC_MODEL_PATH = os.getenv("PIPER_ARABIC_MODEL_PATH", str(BASE_DIR / "models" / "piper" / "ar_JO-kareem-medium.onnx"))
    PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.10"))  # Normal conversational pace (1.10x duration)

    # 2. Cloud Neural Fallback: edge-tts (Microsoft Azure Neural voices — high quality, cloud)
    #    en-US-BrianMultilingualNeural: Standard natural English with support for Arabic loanwords/names (Male).
    #    ar-SA-HamedNeural: Saudi Arabic (Male).
    EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-BrianMultilingualNeural")
    EDGE_TTS_ARABIC_VOICE = os.getenv("EDGE_TTS_ARABIC_VOICE", "ar-SA-HamedNeural")
    EDGE_TTS_RATE = os.getenv("EDGE_TTS_RATE", "+0%")    # Standard natural speed
    EDGE_TTS_VOLUME = os.getenv("EDGE_TTS_VOLUME", "+0%")
    EDGE_TTS_TIMEOUT = float(os.getenv("EDGE_TTS_TIMEOUT", "9.0"))

    # 3. Emergency offline fallback: pyttsx3 (only if edge-tts and piper are unavailable)
    TTS_RATE = int(os.getenv("TTS_RATE", "145"))
    TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))

    # Demo / Mock fallback toggle
    ENABLE_MOCK_FALLBACK = os.getenv("ENABLE_MOCK_FALLBACK", "true").lower() in ("true", "1", "yes")

    @classmethod
    def reload_env(cls):
        """Reload environment variables if changed dynamically."""
        if ENV_DEV_PATH.exists():
            load_dotenv(dotenv_path=ENV_DEV_PATH, override=True)
        cls.PRODUCTS_API = os.getenv("PRODUCTS_API", cls.PRODUCTS_API)
        cls.PRODUCTS_CATEGORY_URL = os.getenv("PRODUCTS_CATEGORY_URL", cls.PRODUCTS_CATEGORY_URL)
        cls.API_TOKEN = os.getenv("API_TOKEN", cls.API_TOKEN)
