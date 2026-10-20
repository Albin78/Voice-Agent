import unittest
import json
from agent import ChefAgent
from cart import OrderCart
from menu_api import MenuAPIClient


class TestRemovalAndOptions(unittest.TestCase):
    def setUp(self):
        self.menu_client = MenuAPIClient()
        self.cart = OrderCart()
        self.agent = ChefAgent(cart=self.cart, menu_client=self.menu_client)

    def test_options_hot_drinks(self):
        """Customer says: 'get me the options of hot drinks' -> must greet and list items with BHD prices, never bare 'Sure'."""
        reply = self.agent.process_turn("get me the options of hot drinks")
        print("\n[TEST OPTIONS HOT DRINKS] Reply:", reply)
        self.assertNotIn(reply.strip().lower(), ["sure", "sure!", "sure.", "certainly", "of course"])
        self.assertTrue("hot drinks" in reply.lower() or "americano" in reply.lower() or "cappuccino" in reply.lower())
        self.assertIn("BHD", reply)
        self.assertIn("?", reply)  # Asks customer which one they would like

    def test_options_cold_drinks(self):
        """Customer says: 'options of cold drinks' -> must list cold drinks with BHD prices."""
        reply = self.agent.process_turn("options of cold drinks")
        print("\n[TEST OPTIONS COLD DRINKS] Reply:", reply)
        self.assertTrue("cold drinks" in reply.lower() or "ice" in reply.lower() or "cold brew" in reply.lower())
        self.assertIn("BHD", reply)

    def test_options_soft_ice(self):
        """Customer says: 'get me the options of soft ice' -> must list soft ice options with BHD prices."""
        reply = self.agent.process_turn("get me the options of soft ice")
        print("\n[TEST OPTIONS SOFT ICE] Reply:", reply)
        self.assertTrue("soft ice" in reply.lower() or "vanilla" in reply.lower())
        self.assertIn("BHD", reply)

    def test_remove_specific_item_from_cart(self):
        """Customer asks to remove a specific item -> item must actually be removed from cart."""
        # Add two items to cart
        self.cart.add_item(101, "Affogato", 1, 2.250)
        self.cart.add_item(102, "Americano Float", 1, 2.250)
        self.assertEqual(len(self.cart.items), 2)
        names = [i.name for i in self.cart.items]
        self.assertIn("Affogato", names)

        # Customer asks to remove Affogato
        reply = self.agent.process_turn("remove the Affogato from my cart")
        print("\n[TEST REMOVE ITEM] Reply:", reply)
        
        # Verify Affogato is removed and ONLY Americano Float remains
        current_names = [i.name for i in self.cart.items]
        self.assertNotIn("Affogato", current_names)
        self.assertIn("Americano Float", current_names)
        self.assertEqual(len(self.cart.items), 1)
        self.assertIn("removed Affogato", reply)

    def test_remove_everything_from_cart(self):
        """Customer asks to remove everything from cart -> cart must actually be completely cleared."""
        self.cart.add_item(101, "Affogato", 1, 2.250)
        self.cart.add_item(102, "Americano Float", 1, 2.250)
        self.cart.add_item(103, "Vanilla Soft Ice", 1, 1.000)
        self.assertEqual(len(self.cart.items), 3)

        reply = self.agent.process_turn("remove everything from cart")
        print("\n[TEST REMOVE EVERYTHING] Reply:", reply)

        self.assertTrue(self.cart.is_empty())
        self.assertEqual(len(self.cart.items), 0)
        self.assertIn("removed everything", reply.lower())

    def test_clear_my_cart(self):
        """Customer says: 'clear my cart' -> cart must be cleared."""
        self.cart.add_item(101, "Cappuccino M", 1, 1.500)
        self.assertFalse(self.cart.is_empty())

        reply = self.agent.process_turn("clear my cart")
        print("\n[TEST CLEAR CART] Reply:", reply)

        self.assertTrue(self.cart.is_empty())

    def test_ordering_is_not_intercepted_by_options(self):
        """Customer says 'get me an Americano Hot M' -> should be an order, not options."""
        # Pre-set cart
        self.assertTrue(self.cart.is_empty())
        # Customer ordering
        reply = self.agent.process_turn("get me an Americano Hot M")
        print("\n[TEST ORDERING NOT OPTIONS] Reply:", reply)
        # Should not treat as options browsing
        self.assertNotIn("Some popular choices are", reply)


if __name__ == "__main__":
    unittest.main()
