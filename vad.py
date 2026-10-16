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
            (
                self.get_speech_timestamps,
                self.save_audio,
                self.read_audio,
                self.VADIterator,
                self.collect_chunks,
            ) = self.utils
            logger.info("Silero VAD v5 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD from torch hub: {e}")
            raise e

    def create_iterator(self):
        return self.VADIterator(
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=30,
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
        vad_iterator = self.create_iterator()
        audio_q = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio buffer status: {status}")
            audio_q.put(indata.copy())

        # Buffer for pre-speech frames (approx 300ms)
        pre_speech_chunks = int(0.3 * self.sample_rate / self.chunk_size)
        pre_speech_ring = collections.deque(maxlen=pre_speech_chunks)

        speech_detected = False
        speech_frames = []
        start_time = time.time()

        # Open non-blocking audio input stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        ):
            logger.info(
                "Microphone stream opened. Listening for speech with Silero VAD"
                " v5..."
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

                # Get speech probability for UI visualization
                with torch.no_grad():
                    speech_prob = self.model(
                        tensor_chunk, self.sample_rate
                    ).item()

                if on_vad_status:
                    status_label = (
                        "speaking" if speech_detected else "listening"
                    )
                    on_vad_status(status_label, speech_prob)

                # Feed chunk to Silero VADIterator
                vad_event = vad_iterator(tensor_chunk)

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
                        logger.info("Silero VAD: Speech end detected.")
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
                                "Max speech length reached (20s). Ending"
                                " recording."
                            )
                            if on_speech_end:
                                on_speech_end()
                            break
                    else:
                        pre_speech_ring.append(chunk_flat)

        if speech_frames:
            full_audio = np.concatenate(speech_frames, axis=0)
            duration = len(full_audio) / self.sample_rate
            logger.info(f"Recorded speech utterance of {duration:.2f} seconds.")
            # Ignore ultra-short spurious noise (<0.3s)
            if duration < 0.3:
                return None
            return full_audio

        return None
