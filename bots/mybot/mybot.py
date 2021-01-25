"""
mybot - mix of parts from the bully bot
"""

# Import the API objects
from api import State
from api import Deck
import random

class Bot:

    def __init__(self):
        pass

    def get_move(self, state):

        # All legal moves
        moves = state.moves()
        chosen_move = moves[0]

        # Get move with highest rank available, of any suit
        for index, move in enumerate(moves):
            if move[0] is not None and move[0] % 5 <= chosen_move[0] % 5:
                chosen_move = move
            else:
                chosen_move = random.choice(moves)

        return chosen_move