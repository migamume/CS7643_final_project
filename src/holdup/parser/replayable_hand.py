import math
import numpy as np
import re

from typing import List, Tuple

from holdup.parser.action_matrix import player_actions_to_matrix, street_int_actions
from holdup.parser.chip_bucket_matrix import street_chip_buckets, chip_buckets_to_matrix
from holdup.parser.hand_matrix import make_hand_matrix
from holdup.parser.player_actions import raw_hand_to_tuple
from holdup.utils import LogProvider

logger = LogProvider.get_logger()

STREETS = {'PREFLOP': 0, 'FLOP': 1, 'TURN': 2, 'RIVER': 3}


"""
Because we want to predict the NEXT action and we train models for each stage,
we only want to train on data where a subsequent, same-street action is available.

For example, the matrix associated with game state after flop has gone check, check
will still be 'on the flop', but there are no future actions available.

To filter out these rows, I see a couple approaches:
1: Define a function that returns a bool whether the street has action remaining
    - Could look at total actions so far to work back... I think

2: Map the entire list of state-by-state actions to a tuple of (actions, street) 
    (This is done now by default in `generate_matrices`)
    - After mapping, filter out the final element and rows prior to street transitions
        ex: 
            (after mapping): [(a1, preflop), (a2, preflop), (a3, flop), (a4, flop), (a5, flop), (a6, flop), (a7, turn), (a8, turn)]
            (after filtering): [(a1, preflop), DELETE, (a3, flop), (a4, flop), (a5, flop), DELETE, (a7, turn), DELETE]
    - Can easiy be done with a functools.reduce(..., game_state_matrices, [[], ""])
    - Or just filtering and dropping the last element on each list
        ex:
        preflop_actions = filter(lambda actions: actions[-1] == STREETS.PREFLOP, a_SINGLE_matrix)
        if len(preflop_actions) == 1:
            # A bot open-folded. No more actions
            preflop_actions = []
        else:
            preflop_actions = preflop_actions[:-1]

        Note that flop,turn,river streets will need an additional check if action list is empty
    
"""
def street_has_future_action(game_state_matrix):
    # TODO
    pass


class ReplayableHand:
    # Note: I did see a hand history file that slightly changed this
    # It had the players first and then STATE:...
    # If we need to use that year's data, it's an easy fix to define a fallback regex
    _raw_hand_regex = "STATE:\d:(\w*\/?\w*\/?\w*\/?\w*):(\w{4})\|(\w{4})(\/?\w*\/?\w*\/?\w*)?:(-?\d*)\|(-?\d*):([^|]*)\|(.*)"

    def __init__(self, raw_hand):
        self.raw_hand = raw_hand

    """
    This could just as easily have been a function instead of a class.
    But I think we may want to add per-hand functionality later. So I'm prematurely optimizing.

    This method maps a raw_hand to N (matrix, round) pairs
    N = total number of actions taken by both players
    and round indicates what street is being played
    """
    def generate_matrices(self) -> List[Tuple[np.ndarray, int]]:
        matches = re.findall(self._raw_hand_regex, self.raw_hand)

        action, player_a_cards, player_b_cards, community_cards, _, _, player_a, player_b = matches[0]
        tupled_hand = raw_hand_to_tuple(action, player_a, player_b)

        street_full_actions = list(map(lambda street: [street[1], street[2]], tupled_hand))

        _street_int_actions = street_int_actions(street_full_actions)
        player_a_actions = [round_actions[0] for round_actions in _street_int_actions]
        player_b_actions = [round_actions[1] for round_actions in _street_int_actions]

        max_rounds = max(len(player_a_actions), len(player_b_actions))

        # Store one matrix per player action
        matrices = []
        
        for round_idx in range(max_rounds):
            for action_idx in range(len(player_a_actions[round_idx]) + len(player_b_actions[round_idx])):
                player_a_end_action_idx = math.floor(action_idx / 2) # 0 => 0; 1 => 0; 2 => 1; 3 => 1
                player_b_end_action_idx = math.ceil(action_idx / 2) # 0 => 0; 1 => 1; 2 => 1; 3 => 2

                partial_a_actions = [
                    player_a_actions[round][0:player_a_end_action_idx+1] if round == round_idx else player_a_actions[round] if round < round_idx else []
                    for round in range(max_rounds)
                ]
                partial_b_actions = [
                    player_b_actions[round][0:player_b_end_action_idx] if round == round_idx else player_b_actions[round] if round < round_idx else []
                    for round in range(max_rounds)
                ]

                both_player_action_matrix = np.vstack(
                    (
                        player_actions_to_matrix(partial_a_actions),
                        player_actions_to_matrix(partial_b_actions)
                    )
                )

                this_round_community_cards = "/".join(community_cards.split('/')[0:round_idx + 1]) if round_idx > 0 else ""

                player_cards = np.vstack((
                    make_hand_matrix(this_round_community_cards, player_a_cards),
                    make_hand_matrix(this_round_community_cards, player_b_cards)
                ))

                partial_street_full_actions = [
                    [
                        street_full_actions[round][player][0:player_a_end_action_idx+1] if player == 0 and round == round_idx else
                        street_full_actions[round][player][0:player_b_end_action_idx] if player == 1 and round == round_idx else
                        street_full_actions[round][player] if player == 0 and round < round_idx else
                        street_full_actions[round][player] if player == 1 and round < round_idx else
                        [] for player in range(2)
                    ]
                    for round in range(max_rounds)
                ]
                scb = street_chip_buckets(partial_street_full_actions)
                chip_bucket_matrix = chip_buckets_to_matrix(scb)

                a_and_b = np.hstack((
                    player_cards,
                    both_player_action_matrix,
                    # Paper incorrectly shows this as 8x4, but it is 8x3
                    # cards.shape = (8, 13); actions.shape = (8, 4)
                    np.zeros((8, 3))
                ))
                a_b_c = np.vstack((
                    a_and_b,
                    chip_bucket_matrix,
                    np.zeros((4, 20))
                ))
                matrices.append((a_b_c, round_idx))

        return matrices
