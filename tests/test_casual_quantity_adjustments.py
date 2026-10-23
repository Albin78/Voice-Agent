import unittest
from agent import ChefAgent
from cart import OrderCart
from menu_api import MenuAPIClient


class TestCasualQuantityAdjustments(unittest.TestCase):
    def setUp(self):
        self.menu_client = MenuAPIClient(force_mock=True)
        self.cart = OrderCart()
        self.agent = ChefAgent(cart=self.cart, menu_client=self.menu_client)

    def test_make_it_2_last_item(self):
        """Customer orders 1 Americano Hot M and then casually says 'make it 2'."""
        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        self.assertEqual(len(self.cart.items), 1)
        self.assertEqual(self.cart.items[0].quantity, 1)
        self.assertEqual(self.cart.get_total(), 1.500)

        reply = self.agent.process_turn("make it 2")
        print("\n[TEST MAKE IT 2] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 3.000)
        self.assertIn("Americano Hot M", reply)
        self.assertIn("2", reply)
        self.assertIn("3.000 BHD", reply)

    def test_make_that_3_word_number(self):
        """Customer says 'make that three'."""
        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        reply = self.agent.process_turn("make that three")
        print("\n[TEST MAKE THAT THREE] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 3)
        self.assertEqual(self.cart.get_total(), 4.500)
        self.assertIn("3", reply)
        self.assertIn("4.500 BHD", reply)

    def test_change_it_to_2(self):
        """Customer says 'change it to 2'."""
        self.cart.add_item(147, "Strawberry Soft Ice", 1, 2.400)
        reply = self.agent.process_turn("change it to 2")
        print("\n[TEST CHANGE IT TO 2] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 4.800)
        self.assertIn("Strawberry Soft Ice", reply)
        self.assertIn("4.800 BHD", reply)

    def test_change_to_2(self):
        """Customer says 'change to 2'."""
        self.cart.add_item(147, "Strawberry Soft Ice", 1, 2.400)
        reply = self.agent.process_turn("change to 2")
        print("\n[TEST CHANGE TO 2] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 4.800)

    def test_add_one_more(self):
        """Customer says 'add one more'."""
        self.cart.add_item(145, "Affogato", 1, 2.700)
        reply = self.agent.process_turn("add one more")
        print("\n[TEST ADD ONE MORE] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 5.400)
        self.assertIn("Affogato", reply)
        self.assertIn("another Affogato", reply)
        self.assertIn("5.400 BHD", reply)

    def test_another_one_please(self):
        """Customer says 'another one please'."""
        self.cart.add_item(130, "Ice Americano", 1, 1.500)
        reply = self.agent.process_turn("another one please")
        print("\n[TEST ANOTHER ONE PLEASE] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 3.000)

    def test_one_more_please(self):
        """Customer says 'one more please'."""
        self.cart.add_item(130, "Ice Americano", 1, 1.500)
        reply = self.agent.process_turn("one more please")
        print("\n[TEST ONE MORE PLEASE] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 3.000)

    def test_give_me_one_more(self):
        """Customer says 'give me one more'."""
        self.cart.add_item(130, "Ice Americano", 1, 1.500)
        reply = self.agent.process_turn("give me one more")
        print("\n[TEST GIVE ME ONE MORE] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)
        self.assertEqual(self.cart.get_total(), 3.000)

    def test_add_2_more(self):
        """Customer says 'add 2 more'."""
        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        reply = self.agent.process_turn("add 2 more")
        print("\n[TEST ADD 2 MORE] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 3)
        self.assertEqual(self.cart.get_total(), 4.500)
        self.assertIn("2 more", reply)

    def test_specific_item_make_the_latte_2(self):
        """When multiple items are in cart, 'make the latte 2' targets the latte."""
        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        self.cart.add_item(117, "Latte Hot M", 1, 2.200)
        self.assertEqual(len(self.cart.items), 2)

        reply = self.agent.process_turn("make the latte 2")
        print("\n[TEST MAKE THE LATTE 2] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 1)  # Americano unchanged
        self.assertEqual(self.cart.items[1].quantity, 2)  # Latte updated to 2
        self.assertEqual(self.cart.get_total(), 1.500 + 4.400)
        self.assertIn("Latte Hot M", reply)

    def test_specific_item_add_another_americano(self):
        """When multiple items are in cart, 'add another Americano' targets Americano."""
        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        self.cart.add_item(147, "Strawberry Soft Ice", 1, 2.400)

        reply = self.agent.process_turn("add another Americano")
        print("\n[TEST ADD ANOTHER AMERICANO] Reply:", reply)

        self.assertEqual(self.cart.items[0].quantity, 2)  # Americano increased to 2
        self.assertEqual(self.cart.items[1].quantity, 1)  # Soft ice unchanged
        self.assertEqual(self.cart.get_total(), 3.000 + 2.400)
        self.assertIn("Americano Hot M", reply)

    def test_empty_cart_casual_quantity(self):
        """When cart is empty, 'make it 2' or 'add one more' informs the customer."""
        self.assertTrue(self.cart.is_empty())
        reply1 = self.agent.process_turn("make it 2")
        self.assertIn("empty", reply1.lower())

        reply2 = self.agent.process_turn("add one more")
        self.assertIn("empty", reply2.lower())

    def test_make_it_zero_removes_item(self):
        """Customer says 'make it 0' -> removes item."""
        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        reply = self.agent.process_turn("make it 0")
        print("\n[TEST MAKE IT 0] Reply:", reply)

        self.assertTrue(self.cart.is_empty())
        self.assertIn("removed", reply.lower())

    def test_tool_dispatcher_update_and_add_more(self):
        """Verify ToolDispatcher directly handles update_quantity and add_more."""
        from tools import ToolDispatcher
        dispatcher = ToolDispatcher(self.menu_client, self.cart)

        self.cart.add_item(115, "Americano Hot M", 1, 1.500)
        res_set = dispatcher.dispatch("update_quantity", {"quantity": 3})
        self.assertIn('"success": true', res_set)
        self.assertEqual(self.cart.items[0].quantity, 3)

        res_add = dispatcher.dispatch("add_more", {"quantity": 2})
        self.assertIn('"success": true', res_add)
        self.assertEqual(self.cart.items[0].quantity, 5)


if __name__ == "__main__":
    unittest.main()
