import io
import re
import logging
from typing import Optional, Callable
import pyttsx3
from config import Config

logger = logging.getLogger("TTS")


def clean_spoken_text(text: str) -> str:
    """
    Sanitize text for spoken audio output:
    Removes markdown formatting, symbols, bullet points, and numbered lists.
    """
    if not text:
        return ""

    # Remove markdown headers (#, ##, etc.)
    text = re.sub(r'#+\s*', '', text)
    # Remove markdown bold/italics (*, **, _, __)
    text = re.sub(r'[\*_]+', '', text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove bullet markers at start of lines (*, -, +)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered list markers (e.g. "1. ", "2. ")
    text = re.sub(r'^\s*\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    # Remove code blocks and inline backticks
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)
    # Remove remaining markdown symbols
    text = re.sub(r'[~|><\^]', '', text)

    # Convert currency e.g. $4.50 to spoken words
    def replace_price(match):
        dollars = match.group(1)
        cents = match.group(2) if match.group(2) else ""
        if cents and int(cents) > 0:
            return f"{dollars} dollars and {int(cents)} cents"
        return f"{dollars} dollars"

    text = re.sub(r'\$(\d+)(?:\.(\d{2}))?', replace_price, text)

    # Normalize multiple whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextToSpeech:
    """
    Persistent Text-To-Speech engine.
    Uses a permanent engine instance to prevent early garbage collection
    (fixing the EspeakDriver ReferenceError and audio cut-off bug).
    Configured with Arabic English accent for Middle Eastern business kiosk use cases.
    """

    def __init__(self, voice_name: str = "Arabic", rate: int = 150):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', 1.0)
        self._set_preferred_voice(voice_name)

    def _set_preferred_voice(self, preferred: str):
        voices = self.engine.getProperty('voices')
        # 1. Exact match
        for v in voices:
            if v.id.lower() == preferred.lower() or v.name.lower() == preferred.lower():
                self.engine.setProperty('voice', v.id)
                logger.info(f"TTS voice set to: {v.name} ({v.id})")
                return

        # 2. Arabic accent match
        for v in voices:
            if "arabic" in v.name.lower() or "arabic" in v.id.lower():
                self.engine.setProperty('voice', v.id)
                logger.info(f"TTS voice set to Arabic accent: {v.name}")
                return

        # 3. Default to first voice
        if voices:
            self.engine.setProperty('voice', voices[0].id)

    def speak(self, text: str, on_start: Optional[Callable[[], None]] = None, on_end: Optional[Callable[[], None]] = None):
        """Clean text and speak completely aloud without cutting off."""
        spoken = clean_spoken_text(text)
        if not spoken:
            return

        logger.info(f"Speaking: '{spoken}'")
        if on_start:
            on_start()

        try:
            # Persistent engine ensures no premature disposal while C audio buffer is playing
            self.engine.say(spoken)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error playing TTS: {e}")
        finally:
            if on_end:
                on_end()
