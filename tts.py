import asyncio
import io
import os
import re
import wave
import logging
from typing import Optional, Callable
from pathlib import Path
import sounddevice as sd
import soundfile as sf
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

    # Convert currency e.g. SAR 15.50 or $15.50 to spoken Riyals and Halalas
    def replace_riyal(match):
        riyals = match.group(1)
        halalas = match.group(2) if match.group(2) else ""
        if halalas and int(halalas) > 0:
            return f"{riyals} riyals and {int(halalas)} halalas"
        return f"{riyals} riyals"

    # Match SAR 15, SAR 15.50, 15 SAR, 15.50 SAR
    text = re.sub(r'SAR\s*(\d+)(?:\.(\d{2}))?', replace_riyal, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)(?:\.(\d{2}))?\s*SAR\b', replace_riyal, text, flags=re.IGNORECASE)
    # Match $15, $15.50
    text = re.sub(r'\$(\d+)(?:\.(\d{2}))?', replace_riyal, text)

    # Normalize multiple whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextToSpeech:
    """
    Natural Text-To-Speech engine for the kiosk.

    Supports:
      1. Edge-TTS (Microsoft Azure Neural voices — high quality, multilingual)
      2. Piper TTS (Local neural ONNX model — Middle Eastern / Arabic accented English, ~120ms latency, zero cloud dependency)
      3. pyttsx3 (Offline legacy fallback — last resort fail-safe)
    """

    def __init__(self):
        self.engine_mode = getattr(Config, "TTS_ENGINE", "edge").lower()
        self.edge_voice = Config.EDGE_TTS_VOICE
        self.edge_rate = Config.EDGE_TTS_RATE
        self.edge_volume = Config.EDGE_TTS_VOLUME

        self.piper_model_path = Config.PIPER_MODEL_PATH
        self.piper_config_path = getattr(
            Config,
            "PIPER_CONFIG_PATH",
            str(Path(self.piper_model_path).with_suffix(".onnx.json")) if self.piper_model_path else ""
        )
        self._piper_voice = None
        self._piper_initialized = False

        self._pyttsx3_engine = None

        if self.engine_mode == "piper":
            # Pre-load Piper voice on startup for 0ms initial latency
            self._get_piper_voice()

    # ------------------------------------------------------------------
    # Piper TTS (Local Neural Voice - Arabic/Middle Eastern accented English)
    # ------------------------------------------------------------------

    def _get_piper_voice(self):
        """Lazy load the Piper ONNX voice once into memory."""
        if self._piper_initialized:
            return self._piper_voice

        self._piper_initialized = True
        try:
            if not os.path.exists(self.piper_model_path):
                logger.warning(f"Piper model not found at {self.piper_model_path}")
                return None

            from piper import PiperVoice
            logger.info(f"Loading Piper neural voice from {self.piper_model_path}...")
            self._piper_voice = PiperVoice.load(
                self.piper_model_path,
                config_path=self.piper_config_path if os.path.exists(self.piper_config_path) else None
            )
            logger.info("Piper neural voice loaded successfully.")
            return self._piper_voice
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            self._piper_voice = None
            return None

    def _speak_piper(self, text: str) -> bool:
        """Synthesize and play speech using Piper neural TTS (100% offline, ~120ms)."""
        try:
            voice = self._get_piper_voice()
            if voice is None:
                return False

            buf = io.BytesIO()
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(length_scale=getattr(Config, "PIPER_LENGTH_SCALE", 1.10))
            with wave.open(buf, 'wb') as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            buf.seek(0)
            data, sr = sf.read(buf)
            sd.play(data, sr)
            sd.wait()
            return True
        except Exception as e:
            logger.warning(f"Piper TTS synthesis failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Edge-TTS (Microsoft Azure Neural)
    # ------------------------------------------------------------------

    def _speak_edge(self, text: str, voice: Optional[str] = None) -> bool:
        """Speak using edge-tts neural voice with configurable timeout."""
        try:
            import edge_tts

            target_voice = voice or self.edge_voice
            timeout_sec = getattr(Config, "EDGE_TTS_TIMEOUT", 9.0)

            async def _async_speak() -> bool:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=target_voice,
                    rate=self.edge_rate,
                    volume=self.edge_volume,
                )
                audio_chunks: list[bytes] = []

                async def _collect():
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_chunks.append(chunk["data"])

                try:
                    await asyncio.wait_for(_collect(), timeout=timeout_sec)
                except (asyncio.TimeoutError, TimeoutError):
                    logger.warning(f"edge-tts stream connection timed out ({timeout_sec:.0f}s).")
                    return False

                if not audio_chunks:
                    return False

                audio_data = b"".join(audio_chunks)
                buf = io.BytesIO(audio_data)
                data, sr = sf.read(buf)
                sd.play(data, sr)
                sd.wait()
                return True

            return asyncio.run(_async_speak())

        except Exception as e:
            logger.warning(f"edge-tts error ({e})")
            return False

    # ------------------------------------------------------------------
    # pyttsx3 (Offline legacy fallback)
    # ------------------------------------------------------------------

    def _speak_pyttsx3(self, text: str) -> None:
        """Offline fallback TTS. Enforces a male voice."""
        try:
            import pyttsx3
            if self._pyttsx3_engine is None:
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty('rate', Config.TTS_RATE)
                self._pyttsx3_engine.setProperty('volume', Config.TTS_VOLUME)
                # Enforce male voice
                try:
                    for v in self._pyttsx3_engine.getProperty('voices'):
                        v_name = (getattr(v, 'name', '') or '').lower()
                        v_id = (getattr(v, 'id', '') or '').lower()
                        v_gender = (str(getattr(v, 'gender', '')) or '').lower()
                        if 'male' in v_gender or 'male' in v_name or 'david' in v_id or 'en-us' in v_id:
                            self._pyttsx3_engine.setProperty('voice', v.id)
                            break
                except Exception:
                    pass
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 fallback error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Clean text and speak it aloud.
        Order of priority:
          1. If TTS_ENGINE == 'piper': Piper TTS -> Edge TTS -> pyttsx3
          2. If TTS_ENGINE == 'edge': Edge TTS -> Piper TTS -> pyttsx3
          3. Default: Edge TTS with seamless Piper TTS fallback to avoid robotic pyttsx3
        """
        spoken = clean_spoken_text(text)
        if not spoken:
            return

        logger.info(f"Speaking text: '{spoken}'")

        if on_start:
            on_start()

        try:
            success = False

            if self.engine_mode == "piper":
                # Primary: Local Piper Neural (Arabic-accented English)
                success = self._speak_piper(spoken)
                if not success:
                    logger.info("Piper TTS unavailable, attempting Edge-TTS...")
                    success = self._speak_edge(spoken)
            else:
                # Primary: Edge-TTS
                success = self._speak_edge(spoken)
                if not success:
                    # High quality local fallback: Piper TTS
                    logger.info("Edge-TTS unavailable or timed out, switching to Piper Neural TTS...")
                    success = self._speak_piper(spoken)

            if not success:
                logger.warning("Neural TTS engines failed, using pyttsx3 as last resort...")
                self._speak_pyttsx3(spoken)

        finally:
            if on_end:
                on_end()
