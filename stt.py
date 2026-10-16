import re
import whisper
import torch
import numpy as np
import logging
from config import Config

logger = logging.getLogger("WhisperSTT")

# Kiosk domain vocabulary prompt to bias Whisper decoding toward restaurant terms
MENU_VOCAB_PROMPT = (
    "Restaurant kiosk order menu: cold drinks, hot drinks, float, kakigori, "
    "shakes, affogato, soft ice, iced americano, iced caramel latte, cappuccino, "
    "cold brew, espresso, chocolate milkshake, vanilla soft ice, muffin, croissant."
)

# Common acoustic/phonetic misrecognitions for accented speech
PHONETIC_CORRECTIONS = [
    (r'\b(?:cord|cold)\s+(?:rings?|ring)\b', 'cold drinks'),
    (r'\b(?:cord|core)\s+(?:drinks?)\b', 'cold drinks'),
    (r'\bsoft\s+eyes?\b', 'soft ice'),
    (r'\bhot\s+rings?\b', 'hot drinks'),
    (r'\bkakiguri\b', 'kakigori'),
]


def post_process_transcription(text: str) -> str:
    """Correct frequent phonetic confusions on accented speech."""
    cleaned = text
    for pattern, replacement in PHONETIC_CORRECTIONS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


class WhisperSTT:
    """Speech-to-Text using OpenAI Whisper with vocabulary priming and phonetic correction."""

    def __init__(self, model_name: str = Config.WHISPER_MODEL, language: str = Config.WHISPER_LANGUAGE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = language
        logger.info(f"Loading Whisper model '{model_name}' on device '{self.device}'...")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text with menu vocabulary priming and domain post-correction.
        """
        if audio is None or len(audio) == 0:
            return ""

        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            use_fp16 = self.device == "cuda"
            result = self.model.transcribe(
                audio,
                language=self.language,
                initial_prompt=MENU_VOCAB_PROMPT,
                fp16=use_fp16,
                temperature=0.0,
                beam_size=5
            )
            raw_text = result.get("text", "").strip()
            
            # Apply phonetic post-processing
            text = post_process_transcription(raw_text)
            logger.info(f"Transcribed: '{raw_text}' -> Post-Processed: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}")
            return ""
