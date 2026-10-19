import re
import whisper
import torch
import numpy as np
import logging
from config import Config

logger = logging.getLogger("WhisperSTT")

# Kiosk domain vocabulary prompt to bias Whisper decoding toward restaurant terms and customizations
MENU_VOCAB_PROMPT = (
    "Restaurant self-service voice kiosk order menu: hot green tea, less sugar, "
    "no sugar, extra sugar, normal sugar, without sugar, low sugar, half sweet, sweet, "
    "condensed sugar, cold drinks, hot drinks, float, kakigori, shakes, affogato, soft ice, "
    "iced americano, iced caramel latte, cappuccino, cold brew, espresso, chocolate milkshake, "
    "vanilla soft ice, muffin, croissant, fresh apple juice, extra ice, no ice, hot, cold, "
    "energy, tired, stressed, relaxing, refreshing, mood recommendation."
)

# Common Whisper hallucinations triggered by silence, breathing, or sighs
HALLUCINATION_PATTERNS = [
    r'\bthank you for watching\b',
    r'\bthanks for watching\b',
    r'\blike,? comment,? and subscribe\b',
    r'\bdon\'?t forget to like\b',
    r'\bsubscribe to my channel\b',
    r'\bsee you in the next video\b',
    r'\bbye[- ]?bye\b',
    r'^(?:auto[.\s,]*)+$',
    r'^(?:you[.\s,]*)+$',
    r'^(?:thank you[.\s,]*)+$',
    r'^\s*you\s*$',
    r'^\s*thank you\s*$',
    r'^\s*\.\s*$',
    r'\b(?:subtitles?|amara\.org|by the amara)\b'
]


# Common acoustic/phonetic misrecognitions for accented speech & customizations
PHONETIC_CORRECTIONS = [
    (r'\b(?:cord|cold)\s+(?:rings?|ring)\b', 'cold drinks'),
    (r'\b(?:cord|core)\s+(?:drinks?)\b', 'cold drinks'),
    (r'\bsoft\s+eyes?\b', 'soft ice'),
    (r'\bhot\s+rings?\b', 'hot drinks'),
    (r'\bkakiguri\b', 'kakigori'),
    (r'\b(?:shooting|shoot\s*in|shot\s*in|suit\s*in|should\s*in)\s+(?:condined|condensed|contained)?\s*sugar\b', 'less sugar'),
    (r'\b(?:shooting|shoot\s*in|shot\s*in|suit\s*in)\s+sugar\b', 'less sugar'),
    (r'\b(?:condined|condence)\s+sugar\b', 'condensed sugar'),
    (r'\b(?:less|las|lass)\s+(?:shoogar|suger|shugar)\b', 'less sugar'),
    (r'\bno\s+(?:shoogar|suger|shugar)\b', 'no sugar'),
    (r'\bgreen\s+(?:item|d|day|tree|tee)\b', 'green tea'),
    (r'\bhot\s+lat\b', 'hot latte'),
    (r'\bcappuchino\b', 'cappuccino'),
]


def is_repetition_hallucination(text: str) -> bool:
    """Detect repetitive single-word or phrase loops from Whisper (e.g. 'Auto. Auto. Auto.')."""
    if not text:
        return False
    words = [w.strip('.,!?;:') for w in text.lower().split() if w.strip('.,!?;:')]
    if not words:
        return False
    # If 3+ words and at most 2 unique words, it's an autoregressive repetition loop
    if len(words) >= 3 and len(set(words)) <= 2:
        return True
    # Regex for consecutive repeated words of 2+ letters
    if re.search(r'\b([a-zA-Z]{2,})(?:[\s.,!?]+\1\b){2,}', text, flags=re.IGNORECASE):
        return True
    return False


def post_process_transcription(text: str) -> str:
    """Correct frequent phonetic confusions on accented speech and filter hallucinations."""
    if not text:
        return ""

    lower_text = text.strip().lower()

    # Reject repetition loops (e.g., "Auto. Auto. Auto. Auto.")
    if is_repetition_hallucination(lower_text):
        logger.info(f"Filtered Whisper repetition hallucination: '{text}'")
        return ""

    for h_pat in HALLUCINATION_PATTERNS:
        if re.search(h_pat, lower_text):
            logger.info(f"Filtered Whisper hallucination: '{text}'")
            return ""

    cleaned = text
    for pattern, replacement in PHONETIC_CORRECTIONS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


class WhisperSTT:
    """Speech-to-Text using OpenAI Whisper with vocabulary priming and domain correction."""

    def __init__(self, model_name: str = Config.WHISPER_MODEL, language: str = Config.WHISPER_LANGUAGE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = language
        logger.info(f"Loading Whisper model '{model_name}' on device '{self.device}'...")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text with menu vocabulary priming, no-speech rejection,
        and domain post-correction.
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
                beam_size=5,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.2,
                no_speech_threshold=0.5,
                logprob_threshold=-1.0
            )

            # Check if Whisper classified this segment as silence/non-speech
            segments = result.get("segments", [])
            if segments:
                avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
                if avg_no_speech > 0.45 or segments[0].get("no_speech_prob", 0.0) > 0.50:
                    logger.info(f"Segment discarded due to high no_speech_prob: avg={avg_no_speech:.2f}")
                    return ""

            raw_text = result.get("text", "").strip()
            
            # Apply phonetic post-processing and hallucination filtering
            text = post_process_transcription(raw_text)
            if text:
                logger.info(f"Transcribed: '{raw_text}' -> Post-Processed: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}")
            return ""
