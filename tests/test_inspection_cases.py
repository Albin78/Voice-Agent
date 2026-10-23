import sys
import os
import unittest
import numpy as np

# Ensure workspace is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from cart import OrderCart
from menu_api import MenuAPIClient
from agent import ChefAgent
from tts import TextToSpeech, clean_spoken_text
from stt import post_process_transcription, MENU_VOCAB_PROMPT

class TestKioskFixes(unittest.TestCase):

    def setUp(self):
        self.cart = OrderCart()
        self.client = MenuAPIClient(force_mock=True)
        self.agent = ChefAgent(cart=self.cart, menu_client=self.client)

    def test_greeting_after_welcome(self):
        """Test that 'hi' or 'hello' after the initial welcome does not loop the welcome speech."""
        welcome = self.agent.get_greeting()
        self.assertIn("welcome", welcome.lower())
        self.assertTrue(self.agent.welcome_given)

        # Customer replies 'hi'
        reply_hi = self.agent.process_turn("hi")
        print(f"Customer: 'hi' -> Chef: '{reply_hi}'")
        # Should NOT repeat the welcome intro
        self.assertNotIn("welcome to girfalco", reply_hi.lower())
        self.assertTrue(any(w in reply_hi.lower() for w in ["hello", "hi", "what can i get", "ready", "help"]))

        # Customer replies 'hi chef'
        reply_chef = self.agent.process_turn("hi chef")
        print(f"Customer: 'hi chef' -> Chef: '{reply_chef}'")
        self.assertNotIn("welcome to girfalco", reply_chef.lower())

    def test_browsing_intent_vs_ordering(self):
        """Test that 'i want to see all items in soft ice' browses products and NEVER adds to cart."""
        self.cart.clear()
        initial_count = len(self.cart.items)
        
        reply = self.agent.process_turn("i want to see all items in soft ice")
        print(f"Customer: 'i want to see all items in soft ice' -> Chef: '{reply}'")
        
        # Verify cart did NOT add items
        self.assertEqual(len(self.cart.items), 0, f"Cart should be empty after browsing! Items: {[i.name for i in self.cart.items]}")
        # Verify reply mentions soft ice items
        self.assertTrue(any(w in reply.lower() for w in ["soft ice", "cone", "cup", "vanilla", "chocolate", "twist"]))

    def test_ordering_intent_adds_item(self):
        """Test that explicit ordering intent 'i want one Vanilla Soft Ice Cone' calls add_item."""
        self.cart.clear()
        reply = self.agent.process_turn("i want one Vanilla Soft Ice Cone please")
        print(f"Customer: 'i want one Vanilla Soft Ice Cone please' -> Chef: '{reply}'")
        
        self.assertGreaterEqual(len(self.cart.items), 1, "Item should have been added to cart!")
        added_item = self.cart.items[0]
        self.assertIn("soft ice", added_item.name.lower())
        self.assertEqual(added_item.quantity, 1)

    def test_tts_speed_configuration(self):
        """Test that TTS speed is calibrated to normal conversational pace."""
        self.assertEqual(Config.EDGE_TTS_RATE, "+0%", "Edge TTS rate should be normal (+0%)")
        self.assertGreaterEqual(getattr(Config, "PIPER_LENGTH_SCALE", 1.0), 1.05, "Piper length scale should be >= 1.05 for normal pace")

    def test_stt_balanced_vocab_prompt(self):
        """Test that STT initial prompt contains conversational and browsing context."""
        from stt import MENU_VOCAB_PROMPT
        prompt_lower = MENU_VOCAB_PROMPT.lower()
        self.assertIn("see", prompt_lower, "STT prompt should include 'see' to prevent acoustic misrecognition as 'order'")
        self.assertIn("soft ice", prompt_lower)
        self.assertIn("hello", prompt_lower)

if __name__ == "__main__":
    unittest.main()
