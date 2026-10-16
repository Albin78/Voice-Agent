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

    # Restaurant Info
    RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "Girfalco")

    # LLM Settings (Ollama)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # Silero VAD Settings
    VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
    # Minimum silence duration in ms before considering speech finished
    VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "800"))
    # Chunk size for Silero VAD v5 at 16kHz
    VAD_CHUNK_SIZE = 512

    # Whisper STT Settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # TTS Settings
    TTS_VOICE = os.getenv("TTS_VOICE", "Arabic")
    TTS_RATE = int(os.getenv("TTS_RATE", "150"))
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
