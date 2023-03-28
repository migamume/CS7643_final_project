from enum import IntEnum
import math
import numpy as np
import re

from typing import Dict, List,  Tuple

from holdup.parser.action_matrix import player_actions_to_matrix, street_int_actions
from holdup.parser.chip_bucket_matrix import street_chip_buckets, chip_buckets_to_matrix
from holdup.parser.hand_matrix import make_hand_matrix
from holdup.parser.player_actions import raw_hand_to_tuple
from holdup.utils import LogProvider

logger = LogProvider.get_logger()

class Streets(IntEnum):
    NextHand = -1
    Preflop = 0
    Flop = 1
    Turn = 2
    River = 3

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

    This method maps a raw_hand to a dict of Round keys and (matrix, (next_action_crosses_round_boundary, next_action)) pairs
        Note: the final action has a special synax: (next_action_crosses_round_boundary: true, next_action: -1)
        This indicates there is no new action as we would require a new hand
    N = total number of actions taken by both players
    and round indicates what street is being played
    """
    def matrix_next_action_by_street(self) -> Dict[int, List[Tuple[np.ndarray, Tuple[bool, int]]]]:
        matches = re.findall(self._raw_hand_regex, self.raw_hand)

        action, player_a_cards, player_b_cards, community_cards, _, _, player_a, player_b = matches[0]
        tupled_hand = raw_hand_to_tuple(action, player_a, player_b)

        street_full_actions = list(map(lambda street: [street[1], street[2]], tupled_hand))

        _street_int_actions = street_int_actions(street_full_actions)
        player_a_actions = [round_actions[0] for round_actions in _street_int_actions]
        player_b_actions = [round_actions[1] for round_actions in _street_int_actions]

        max_rounds = max(len(player_a_actions), len(player_b_actions))

        actions_by_round = {street: [] for street in range(int(Streets.Preflop), int(Streets.River) + 1)}

        for round_idx in range(max_rounds):
            num_actions_this_round = len(player_a_actions[round_idx]) + len(player_b_actions[round_idx])
            for action_idx in range(num_actions_this_round):
                player_a_current_action_end_idx = math.floor(action_idx / 2) # 0 => 0; 1 => 0; 2 => 1; 3 => 1
                player_b_current_action_end_idx = math.ceil(action_idx / 2) # 0 => 0; 1 => 1; 2 => 1; 3 => 2

                partial_a_actions = [
                    player_a_actions[round][0:player_a_current_action_end_idx+1] if round == round_idx else player_a_actions[round] if round < round_idx else []
                    for round in range(max_rounds)
                ]
                partial_b_actions = [
                    player_b_actions[round][0:player_b_current_action_end_idx] if round == round_idx else player_b_actions[round] if round < round_idx else []
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
                        street_full_actions[round][player][0:player_a_current_action_end_idx+1] if player == 0 and round == round_idx else
                        street_full_actions[round][player][0:player_b_current_action_end_idx] if player == 1 and round == round_idx else
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

                # Remember that range(n) iterates from [0-n)
                next_action_crosses_round_boundary = action_idx == num_actions_this_round - 1

                if not next_action_crosses_round_boundary:
                    # action_idx 0 implies the next player is player_b
                    if action_idx % 2 == 0:
                        next_action = player_b_actions[round_idx][player_b_current_action_end_idx]
                    else:
                        next_action = player_a_actions[round_idx][player_a_current_action_end_idx + 1]
                else:
                    is_last_round = round_idx == max_rounds - 1
                    if is_last_round:
                        next_action = -1
                    else:
                        # A new round always starts with player_a
                        next_action = player_a_actions[round_idx + 1][0]

                actions_by_round[round_idx].append((a_b_c, (next_action_crosses_round_boundary, next_action)))

        return actions_by_round
