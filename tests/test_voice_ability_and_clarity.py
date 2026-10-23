import unittest
import numpy as np
import soundfile as sf
from scipy import signal

from tts import TextToSpeech, clean_spoken_text, split_into_speech_chunks
from agent import ChefAgent, condense_long_spoken_list
from cart import OrderCart
from menu_api import MenuAPIClient
from stt import WhisperSTT


class TestVoiceAbilityAndClarity(unittest.TestCase):
    """
    Test suite verifying voice ability, phonetic clarity, audio chunking,
    and concise conversational behavior for long menu texts.
    """

    @classmethod
    def setUpClass(cls):
        cls.tts = TextToSpeech()
        cls.stt = WhisperSTT()

    def setUp(self):
        self.cart = OrderCart()
        self.menu_client = MenuAPIClient(force_mock=True)
        self.agent = ChefAgent(cart=self.cart, menu_client=self.menu_client)

    def test_split_into_speech_chunks_exact_user_sentence(self):
        """Verify that a long 9-item sentence is broken down into breath units with pauses."""
        long_text = (
            "We also have Ice Americano for 1.5 BHD, Ice Caramel Latte for 2.7 BHD, "
            "Ice Chocolate for 2.7 BHD, Ice Hojicha Latte for 2.7 BHD, Ice Latte for 2.7 BHD, "
            "Lemon Mojito for 2.6 BHD, Passion Fruit Mojito for 2.6 BHD, "
            "Strawberry Mojito for 2.6 BHD, and V60 Cold again at 2.4 BHD. "
            "Which one would you like?"
        )
        chunks = split_into_speech_chunks(long_text, max_chunk_chars=90)
        
        # Must split into multiple conversational chunks (not a giant blob)
        self.assertGreaterEqual(len(chunks), 5)
        
        # Each chunk must be reasonably short (< 100 chars)
        for chunk_text, pause_ms in chunks:
            self.assertLess(len(chunk_text), 100)
            self.assertIn(pause_ms, [140, 250])

        # Last chunk should be the question
        self.assertIn("Which one would you like", chunks[-1][0])
        self.assertEqual(chunks[-1][1], 250)

    def test_split_into_speech_chunks_short_sentence(self):
        """Short sentences should remain intact with natural 250ms cadence pause."""
        text = "Hi, welcome to Girfalco! What can I get for you today?"
        chunks = split_into_speech_chunks(text, max_chunk_chars=90)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0], "Hi, welcome to Girfalco!")
        self.assertEqual(chunks[1][0], "What can I get for you today?")

    def test_clean_spoken_text_bhd_and_menu_phonetics(self):
        """Verify BHD currency decimal normalization and V60 pronunciation."""
        raw = "Ice Americano for 1.500 BHD, V60 Cold for 2.400 BHD, and Affogato for 2.000 BHD."
        cleaned = clean_spoken_text(raw)
        
        # Normalizes trailing zeros: 1.500 -> 1.5, 2.400 -> 2.4, 2.000 -> 2
        self.assertIn("1.5 BHD", cleaned)
        self.assertIn("2.4 BHD", cleaned)
        self.assertIn("2 BHD", cleaned)
        self.assertNotIn("1.500", cleaned)
        self.assertNotIn("2.400", cleaned)
        self.assertNotIn("2.000", cleaned)

        # V60 is written as V sixty for clear speech
        self.assertIn("V sixty", cleaned)

    def test_synthesize_audio_clarity_with_whisper(self):
        """
        Synthesize the long multi-item speech using the new chunked TTS engine,
        then feed it into Whisper STT to verify high intelligibility (>90% item recall).
        """
        test_text = (
            "We also have Ice Americano for 1.5 BHD, Ice Caramel Latte for 2.7 BHD, "
            "Ice Chocolate for 2.7 BHD, Ice Latte for 2.7 BHD, Lemon Mojito for 2.6 BHD, "
            "Strawberry Mojito for 2.6 BHD, and V60 Cold again at 2.4 BHD. "
            "Which one would you like?"
        )
        
        res = self.tts.synthesize_audio(test_text)
        self.assertIsNotNone(res, "Audio synthesis must succeed")
        audio_data, sample_rate = res

        self.assertGreater(len(audio_data), 0)
        self.assertGreater(sample_rate, 8000)

        # Resample to 16kHz for Whisper
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_16k = signal.resample(audio_data, num_samples).astype(np.float32)

        transcription = self.stt.transcribe(audio_16k).lower()
        print("\n[WHISPER TRANSCRIPTION OF SYNTHESIZED LONG TEXT]:\n", transcription)

        # Verify key menu items and numbers were clearly synthesized and heard
        self.assertTrue("americano" in transcription, "Americano must be clearly audible")
        self.assertTrue("caramel latte" in transcription or "latte" in transcription, "Latte must be audible")
        self.assertTrue("chocolate" in transcription, "Chocolate must be audible")
        self.assertTrue("mojito" in transcription, "Mojito must be audible")
        self.assertTrue("v60" in transcription or "sixty" in transcription, "V60 must be audible")
        self.assertTrue("which one would you like" in transcription, "Ending question must be audible")

    def test_customer_asks_other_cold_drinks_available(self):
        """
        Customer: 'What are the other cold drinks available?'
        Agent must respond with a warm, voice-friendly answer:
        - At most 3 popular highlighted choices with BHD prices
        - Informs customer that more options are on the screen
        - Never overloads customer with a 9-item wall of text
        """
        reply = self.agent.process_turn("What are the other cold drinks available?")
        print("\n[CHEF REPLY TO 'What are the other cold drinks available?']:\n", reply)

        self.assertIn("BHD", reply)
        self.assertTrue("cold drinks" in reply.lower() or "ice" in reply.lower())
        self.assertIn("?", reply)
        
        # Verify it lists items with screen notice
        self.assertTrue("screen" in reply.lower() or "more" in reply.lower())

        # Count occurrences of 'for X BHD' in the response — must NOT exceed 3 items
        import re
        bhd_matches = re.findall(r'\b\d+(?:\.\d+)?\s*BHD\b', reply)
        self.assertLessEqual(len(bhd_matches), 3, f"Agent should speak at most 3 items over voice, got {len(bhd_matches)}")

    def test_condense_long_spoken_list(self):
        """Verify condense_long_spoken_list condenses runaway 9-item enumerations."""
        raw_9_items = (
            "We also have Ice Americano for 1.5 BHD, Ice Caramel Latte for 2.7 BHD, "
            "Ice Chocolate for 2.7 BHD, Ice Hojicha Latte for 2.7 BHD, Ice Latte for 2.7 BHD, "
            "Lemon Mojito for 2.6 BHD, Passion Fruit Mojito for 2.6 BHD, "
            "Strawberry Mojito for 2.6 BHD, and V60 Cold again at 2.4 BHD. "
            "Which one would you like?"
        )
        condensed = condense_long_spoken_list(raw_9_items)
        print("\n[CONDENSED LONG SPOKEN LIST]:\n", condensed)

        # Must condense to 3 items
        self.assertIn("Ice Americano for 1.5 BHD", condensed)
        self.assertIn("Ice Caramel Latte for 2.7 BHD", condensed)
        self.assertIn("Ice Chocolate for 2.7 BHD", condensed)
        self.assertIn("plus more options on the screen", condensed)
        self.assertIn("Which one would you like?", condensed)

        # Must NOT contain the 4th through 9th items
        self.assertNotIn("Hojicha", condensed)
        self.assertNotIn("Lemon Mojito", condensed)
        self.assertNotIn("Strawberry Mojito", condensed)


if __name__ == "__main__":
    unittest.main()
