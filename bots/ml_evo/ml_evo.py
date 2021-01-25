#!/usr/bin/env python

from api import State, util
from api import Deck
import random, os
from itertools import chain
import random
from bots.rdeep import rdeep
from numpy import random  #### added
from scipy.stats import bernoulli  ####added

import joblib

# Path of the model we will use. If you make a model
# with a different name, point this line to its path.
DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/bully_model.pkl' # Change to model you want to use

class Bot:
    __randomize = True  # then random shuffles moves

    __model = None

    def __init__(self, randomize=True, model_file=DEFAULT_MODEL, theta=[0.2626715,0.6176881]):  # Change to optimal theta values recieved from Evo_optimizer

        print(model_file)
        self.__randomize = randomize
        self.theta = theta
        self.dm_rdeep = rdeep.Bot()

        # Load the model
        self.__model = joblib.load(model_file)

    def bully_move(self, state):  #### Here bully strategy is implemented/starts
        # type: (State) -> tuple[int, int]
        # """
        # Function that gets called every turn. This is where to implement the strategies.
        # Be sure to make a legal move. Illegal moves, like giving an index of a card you
        # don't own or proposing an illegal mariage, will lose you the game.
        # TODO: add some more explanation
        # :param State state: An object representing the gamestate. This includes a link to
        # 	the states of all the cards, the trick and the points.
        # :return: A tuple of integers or a tuple of an integer and None,
        # 	indicating a move; the first indicates the card played in the trick, the second a
        # 	potential spouse.
        # """
        # All legal moves

        moves = state.moves()
        chosen_move = moves[0]

        moves_trump_suit = []

        # Get all trump suit moves available
        for index, move in enumerate(moves):

            if move[0] is not None and Deck.get_suit(move[0]) == state.get_trump_suit():
                moves_trump_suit.append(move)

        if len(moves_trump_suit) > 0:
            chosen_move = moves_trump_suit[0]
            return chosen_move

        # If the opponent has played a card
        if state.get_opponents_played_card() is not None:

            moves_same_suit = []

            # Get all moves of the same suit as the opponent's played card
            for index, move in enumerate(moves):
                if move[0] is not None and Deck.get_suit(move[0]) == Deck.get_suit(state.get_opponents_played_card()):
                    moves_same_suit.append(move)

            if len(moves_same_suit) > 0:
                chosen_move = moves_same_suit[0]
                return chosen_move

            # Get move with highest rank available, of any suit
        for index, move in enumerate(moves):
            if move[0] is not None and move[0] % 5 <= chosen_move[0] % 5:
                chosen_move = move

        return chosen_move  #### Here bully strategy ends

    #### Here other strategies are implemented

    def get_move(self, state):  ##### This function will have to be modified when adding more strategies

        val, move = self.value(state)
        move_bully = self.bully_move(state)
        move_rdeep = self.dm_rdeep.get_move(state)
        rdeep_score = self.dm_rdeep.chance


        if bernoulli(p=(self.theta[1] / 10)) == 1:  # bernoulli = generates a random number distribution
            return move_bully
        else:
            if val * self.theta[0] > rdeep_score * (1 - self.theta[0]):  #### This if-statement will have to be modified when adding more strategies
                return move
            else:
                return move_rdeep

        # if val * self.theta[0] > rdeep_score * (1 - self.theta[0]): #### This if-statement will have to be modified when adding more strategies
        #     return move
        # elif val * self.theta[0] < rdeep_score * (1 - self.theta[0]):
        #     return rdeep_move
        # else:
        #     return move_bully

    def value(self, state):
        """
        Return the value of this state and the associated move
        :param state:
        :return: val, move: the value of the state, and the best move.
        """

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a function call so that 'value' will
            # contain the predicted value of 'next_state'
            # NOTE: This is different from the line in the minimax/alphabeta bot
            value = self.heuristic(next_state)  # maximizing(next_state)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_value, best_move

    def heuristic(self, state):

        # Convert the state to a feature vector
        feature_vector = [features(state)]

        # These are the classes: ('won', 'lost')
        classes = list(self.__model.classes_)

        # Ask the model for a prediction
        # This returns a probability for each class
        prob = self.__model.predict_proba(feature_vector)[0]

        # Weigh the win/loss outcomes (-1 and 1) by their probabilities
        res = -1.0 * prob[classes.index('lost')] + 1.0 * prob[classes.index('won')]

        return res


def maximizing(state):
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1


def get_num_trump_cards(trump_suit, hand):
    num_trump_cards = 0
    for card in hand:
        # print("CARD: " + str(card))
        index = int(card / 5)
        # print("INDEX: " + str(index))
        if (index == 3 and trump_suit == "S") or (index == 2 and trump_suit == "H") or (
                index == 1 and trump_suit == "D") or (index == 0 and trump_suit == "C"):
            num_trump_cards += 1
    # print("NUM_TRUMP_CARDS: " + str(num_trump_cards))
    return num_trump_cards


def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.

    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    feature_set = []

    # Add player 1's points to feature set
    p1_points = state.get_points(1)

    # Add player 2's points to feature set
    p2_points = state.get_points(2)

    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(1)

    # Add plauer 2's pending points to feature set
    p2_pending_points = state.get_pending_points(2)

    # Get trump suit
    trump_suit = state.get_trump_suit()

    # Add phase to feature set
    phase = state.get_phase()

    # Add stock size to feature set
    stock_size = state.get_stock_size()

    # Add leader to feature set
    leader = state.leader()

    # Add final game points to the feature set
    final = state.winner()
    game_points = final[1]

    # Add whose turn it is to feature set
    whose_turn = state.whose_turn()

    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()

    # Add the number of trump cards in player's hand
    num_trump_cards = get_num_trump_cards(state.get_trump_suit(), state.hand())

    ################## You do not need to do anything below this line ########################

    perspective = state.get_perspective()

    # Perform one-hot encoding on the perspective.
    # Learn more about one-hot here: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    perspective = [card if card != 'U' else [1, 0, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'S' else [0, 1, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P1H' else [0, 0, 1, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P2H' else [0, 0, 0, 1, 0, 0] for card in perspective]
    perspective = [card if card != 'P1W' else [0, 0, 0, 0, 1, 0] for card in perspective]
    perspective = [card if card != 'P2W' else [0, 0, 0, 0, 0, 1] for card in perspective]

    # Append one-hot encoded perspective to feature_set
    feature_set += list(chain(*perspective))

    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points / total_points if total_points > 0 else 0.)
    feature_set.append(p2_points / total_points if total_points > 0 else 0.)

    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points / total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points / total_pending_points if total_pending_points > 0 else 0.)

    # Convert trump suit to id and add to feature set
    # You don't need to add anything to this part
    suits = ["C", "D", "H", "S"]
    trump_suit_onehot = [0, 0, 0, 0]
    trump_suit_onehot[suits.index(trump_suit)] = 1
    feature_set += trump_suit_onehot

    # Append one-hot encoded phase to feature set
    feature_set += [1, 0] if phase == 1 else [0, 1]

    # Append normalized stock size to feature set
    feature_set.append(stock_size / 10)

    # Append one-hot encoded leader to feature set
    feature_set += [1, 0] if leader == 1 else [0, 1]

    # Append one-hot encoded whose_turn to feature set
    feature_set += [1, 0] if whose_turn == 1 else [0, 1]

    # Append one-hot encoded opponent's card to feature set
    opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
    feature_set += opponents_played_card_onehot

    # Return feature set
    return feature_set
