from unittest import TestCase
from api import Deck

class TestDeck(TestCase):

    def test_generate(self):
        d = Deck.generate(0)

        stock = d.get_card_states().count("S")
        self.assertEqual(stock,10,"The value should be 10")

        player1 = d.get_card_states().count("P1H")
        self.assertEqual(player1, 5,"The value should be 5")

        player2 = d.get_card_states().count("P2H")
        self.assertEqual(player2, 5,"The value should be 5")

    def test_trump_exchange(self):

		d = Deck.generate(0)
		print(d.get_trump_suit()) #added a bracket around the print-statement
		print(d.get_card_states()) #added a bracket around the print-statement
		if d.can_exchange(1):
			print("1 exchanged")	#added a bracket around the print-statement
			d.exchange_trump()
			print(d.get_card_states())	#added a bracket around the print-statement

		elif d.can_exchange(2):
			print("2 exchanged")	#added a bracket around the print-statement
			d.exchange_trump()
			print(d.get_card_states())	#added a bracket around the print-statement
		else:
			print("no one exchanged")	#added a bracket around the print-statement