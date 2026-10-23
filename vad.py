import collections
import logging
import queue
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd
import torch
from config import Config

logger = logging.getLogger("SileroVAD")


class KioskVADIterator:
    """
    High-performance streaming VAD iterator for restaurant kiosks.
    Performs exactly ONE forward pass per audio chunk (preventing state corruption),
    tracks speech probability for UI soundwaves, and allows configurable pause tolerance
    (default 1400ms) so customers are never cut off while hesitating or thinking mid-sentence.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.50,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 1400,
        speech_pad_ms: int = 250,
        neg_threshold: Optional[float] = None,
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)
        self.neg_threshold = neg_threshold

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.last_speech_prob = 0.0
        self.reset_states()

    def reset_states(self):
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()
        self.last_speech_prob = speech_prob

        # If voice resumes after a pause, reset the silence countdown
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        # Speech onset detected
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {"start": int(speech_start)}

        # Speech end candidate
        if (speech_prob < self.neg_threshold) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {"end": int(speech_end)}

        return None


class SileroVADDetector:
    """Real-time Voice Activity Detection using Silero VAD v5 locally."""

    def __init__(
        self,
        sample_rate: int = Config.VAD_SAMPLE_RATE,
        threshold: float = Config.VAD_THRESHOLD,
        min_silence_ms: int = Config.VAD_MIN_SILENCE_MS,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.chunk_size = Config.VAD_CHUNK_SIZE  # 512 samples for 16kHz

        logger.info("Loading Silero VAD v5 model locally...")
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                onnx=False,
            )
            logger.info("Silero VAD v5 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD from torch hub: {e}")
            raise e

        self._has_mic = self._detect_mic()

    @staticmethod
    def _detect_mic() -> bool:
        """Return True if the host has a valid audio input device (microphone)."""
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            return any(d.get("max_input_channels", 0) > 0 for d in devs)
        except Exception:
            return False

    def create_iterator(self):
        return KioskVADIterator(
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=getattr(Config, "VAD_SPEECH_PAD_MS", 250),
        )

    def listen_and_record_speech(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_vad_status: Optional[Callable[[str, float], None]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Listens to the microphone using Silero VAD v5.

        Accumulates audio frames while speech is detected.
        Returns the recorded audio as a float32 numpy array once the user
        finishes speaking.
        """
        if not getattr(self, "_has_mic", True):
            logger.info("Host has no physical microphone device (headless VM). Skipping hardware listen.")
            return None

        vad_iterator = self.create_iterator()
        audio_q = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio buffer status: {status}")
            audio_q.put(indata.copy())

        # Buffer for pre-speech frames (approx 400ms to preserve initial syllables)
        pre_speech_chunks = int(0.4 * self.sample_rate / self.chunk_size)
        pre_speech_ring = collections.deque(maxlen=pre_speech_chunks)

        speech_detected = False
        speech_frames = []
        start_time = time.time()

        # Open non-blocking audio input stream
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=audio_callback,
            ):
                logger.info(
                    "Microphone stream opened. Listening for speech with Silero VAD v5..."
                )
                if on_vad_status:
                    on_vad_status("listening", 0.0)

                while True:
                    if timeout and (time.time() - start_time) > timeout:
                        if not speech_detected:
                            return None

                    try:
                        chunk = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    chunk_flat = chunk.flatten()
                    tensor_chunk = torch.from_numpy(chunk_flat)

                    # Feed chunk to single-pass KioskVADIterator (advances RNN state once)
                    vad_event = vad_iterator(tensor_chunk)
                    speech_prob = vad_iterator.last_speech_prob

                    if on_vad_status:
                        status_label = (
                            "speaking" if speech_detected else "listening"
                        )
                        on_vad_status(status_label, speech_prob)

                    if vad_event is not None:
                        if "start" in vad_event and not speech_detected:
                            speech_detected = True
                            logger.info("Silero VAD: Speech start detected!")
                            if on_speech_start:
                                on_speech_start()
                            # Prepend buffered audio so word beginnings are not clipped
                            speech_frames.extend(list(pre_speech_ring))
                            speech_frames.append(chunk_flat)

                        elif "end" in vad_event and speech_detected:
                            logger.info("Silero VAD: Speech end detected after silence hold.")
                            speech_frames.append(chunk_flat)
                            if on_speech_end:
                                on_speech_end()
                            break
                    else:
                        if speech_detected:
                            speech_frames.append(chunk_flat)
                            # Safety check: prevent runaway recording (>20 seconds)
                            if (
                                len(speech_frames) * self.chunk_size
                            ) / self.sample_rate > 20:
                                logger.info(
                                    "Max speech length reached (20s). Ending recording."
                                )
                                if on_speech_end:
                                    on_speech_end()
                                break
                        else:
                            pre_speech_ring.append(chunk_flat)
        except (sd.PortAudioError, Exception) as e:
            logger.warning(f"Hardware audio device unavailable ({e}). Host appears headless/VM.")
            return None


        if speech_frames:
            full_audio = np.concatenate(speech_frames, axis=0)
            duration = len(full_audio) / self.sample_rate

            # Ignore ultra-short clicks/pops (<0.28s), but preserve natural single-syllable greetings ('hi', 'hey', 'yes')
            if duration < 0.28:
                return None

            # Calculate RMS energy: breathing is typically < 0.012, spoken voice > 0.025
            rms = float(np.sqrt(np.mean(full_audio**2)))
            logger.info(f"Recorded utterance: {duration:.2f}s with RMS energy: {rms:.4f}")
            if rms < 0.014:
                logger.info(f"Utterance energy too low ({rms:.4f} < 0.014). Discarding breathing/ambient noise.")
                return None

            # Spectral Flatness Measure (SFM):
            # Speech has tonal peaks -> SFM < 0.55.
            # Broadband noise / HVAC / music -> SFM close to 1.0.
            # Threshold of 0.72 is conservative - catches clear noise without rejecting accented speech.
            fft_power = np.abs(np.fft.rfft(full_audio)) ** 2 + 1e-12
            sfm = float(np.exp(np.mean(np.log(fft_power))) / np.mean(fft_power))
            if sfm > 0.72:
                logger.info(f"High spectral flatness ({sfm:.3f} > 0.72) - likely broadband noise. Discarding.")
                return None

            return full_audio

        return None


# Backward-compatibility alias
VoiceActivityDetector = SileroVADDetector
