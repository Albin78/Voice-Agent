import unittest
from agent import ChefAgent
from cart import OrderCart
from menu_api import MenuAPIClient


class TestOrderPlacementFlow(unittest.TestCase):
    def setUp(self):
        self.menu_client = MenuAPIClient(force_mock=True)
        self.cart = OrderCart()
        self.agent = ChefAgent(cart=self.cart, menu_client=self.menu_client)

    def test_full_add_remove_and_place_order_flow(self):
        """
        Tests complete customer journey:
        1. Customer adds 2 items (Americano Hot M: 1.500 BHD, Vanilla Soft Ice: 1.000 BHD, Affogato: 2.250 BHD).
        2. Customer removes Affogato.
        3. Customer says 'place the order' -> Agent MUST ask for confirmation with items and total (2.500 BHD).
        4. Customer confirms with 'yes' -> Order placed, order number generated, cart emptied, last_placed_order preserved.
        """
        # Step 1: Add items
        self.cart.add_item(101, "Americano Hot M", 1, 1.500)
        self.cart.add_item(102, "Vanilla Soft Ice", 1, 1.000)
        self.cart.add_item(103, "Affogato", 1, 2.250)
        self.assertEqual(len(self.cart.items), 3)
        self.assertEqual(self.cart.get_total(), 4.750)

        # Step 2: Remove Affogato
        rem_reply = self.agent.process_turn("remove the Affogato from my cart")
        print("\n[STEP 2: REMOVE AFFOGATO] Reply:", rem_reply)
        self.assertEqual(len(self.cart.items), 2)
        self.assertEqual(self.cart.get_total(), 2.500)
        self.assertNotIn("Affogato", [i.name for i in self.cart.items])

        # Step 3: Customer says 'place the order'
        confirm_ask_reply = self.agent.process_turn("place the order")
        print("\n[STEP 3: PLACE ORDER (ASK CONFIRMATION)] Reply:", confirm_ask_reply)
        
        # Verify Chef asks confirmation with items and total amount
        self.assertTrue(self.cart.confirmation_requested)
        self.assertFalse(self.cart.is_submitted)
        self.assertIn("Americano Hot M", confirm_ask_reply)
        self.assertIn("Vanilla Soft Ice", confirm_ask_reply)
        self.assertIn("2.500 BHD", confirm_ask_reply)
        self.assertIn("Shall I place this order?", confirm_ask_reply)

        # Step 4: Customer confirms with 'yes'
        placed_reply = self.agent.process_turn("yes")
        print("\n[STEP 4: CUSTOMER CONFIRMS (YES)] Reply:", placed_reply)

        # Verify order is finalized
        self.assertTrue(self.cart.is_submitted)
        self.assertIsNotNone(self.cart.order_number)
        self.assertIn(self.cart.order_number, placed_reply)
        self.assertIn("2.500 BHD", placed_reply)
        self.assertIn("Please proceed to the counter", placed_reply)

        # Verify last_placed_order has full details
        placed_info = self.cart.last_placed_order
        self.assertIsNotNone(placed_info)
        self.assertEqual(placed_info["order_number"], self.cart.order_number)
        self.assertEqual(placed_info["total"], 2.500)
        self.assertEqual(len(placed_info["items"]), 2)

        # Verify cart items are emptied only AFTER order is placed
        self.assertEqual(len(self.cart.items), 0)
        self.assertTrue(self.cart.is_empty())

    def test_place_order_empty_cart(self):
        """Customer says 'place the order' on empty cart -> Chef informs tray is empty."""
        self.assertTrue(self.cart.is_empty())
        reply = self.agent.process_turn("place the order")
        print("\n[TEST EMPTY CART PLACE] Reply:", reply)
        self.assertIn("empty", reply.lower())
        self.assertFalse(self.cart.confirmation_requested)

    def test_customer_declines_confirmation(self):
        """Customer says 'no' when asked for confirmation -> Chef offers to change order."""
        self.cart.add_item(101, "Americano Hot M", 1, 1.500)
        self.agent.process_turn("place the order")
        self.assertTrue(self.cart.confirmation_requested)

        decline_reply = self.agent.process_turn("no, not yet")
        print("\n[TEST DECLINE CONFIRMATION] Reply:", decline_reply)
        self.assertFalse(self.cart.confirmation_requested)
        self.assertFalse(self.cart.is_submitted)
        self.assertIn("change", decline_reply.lower())
        self.assertEqual(len(self.cart.items), 1)

    def test_consecutive_place_the_order_places_order(self):
        """
        When asked 'place the order', Chef asks for confirmation with total price.
        Next, when customer asks 'place the order' again, the order is placed without confusing the consumer.
        """
        self.cart.add_item(101, "Ice Americano", 1, 1.500)
        self.cart.add_item(102, "Ice Caramel Latte", 1, 2.700)
        self.assertEqual(self.cart.get_total(), 4.200)

        # Turn 1: Customer asks 'place the order'
        r1 = self.agent.process_turn("place the order")
        print("\n[CONSECUTIVE TURN 1] Chef:", r1)
        self.assertTrue(self.cart.confirmation_requested)
        self.assertFalse(self.cart.is_submitted)
        self.assertIn("Ice Americano", r1)
        self.assertIn("Ice Caramel Latte", r1)
        self.assertIn("4.200 BHD", r1)
        self.assertIn("Shall I place this order?", r1)

        # Turn 2: Customer repeats 'place the order'
        r2 = self.agent.process_turn("place the order")
        print("\n[CONSECUTIVE TURN 2] Chef:", r2)
        self.assertTrue(self.cart.is_submitted)
        self.assertIsNotNone(self.cart.order_number)
        self.assertIn(self.cart.order_number, r2)
        self.assertIn("4.200 BHD", r2)
        self.assertIn("Please proceed to the counter", r2)
        self.assertTrue(self.cart.is_empty())

    def test_turn2_place_order_variations(self):
        """Test variations like 'place this order', 'place that order', 'can you place the order' on Turn 2."""
        variations = [
            "place this order",
            "place that order",
            "can you place the order",
            "please place the order",
            "place it"
        ]
        for phrase in variations:
            cart = OrderCart()
            agent = ChefAgent(cart=cart, menu_client=self.menu_client)
            cart.add_item(101, "Americano Hot M", 1, 1.500)

            # Turn 1: asks confirmation with total
            r1 = agent.process_turn("place the order")
            self.assertTrue(cart.confirmation_requested)
            self.assertIn("1.500 BHD", r1)

            # Turn 2: customer confirms using variation phrase
            r2 = agent.process_turn(phrase)
            self.assertTrue(cart.is_submitted, f"Failed to submit on phrase: '{phrase}'")
            self.assertIn("placed for", r2)
            self.assertTrue(cart.is_empty())


if __name__ == "__main__":
    unittest.main()
