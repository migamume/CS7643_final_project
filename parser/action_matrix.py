import numpy as np

from parser.parser import CALL, CHECK, FOLD, RAISE


action_to_int_mappings = {CHECK: 0, CALL: 1, RAISE: 2, FOLD: 3}
full_action_to_int = lambda full_action: action_to_int_mappings[full_action[0]]

def street_int_actions(full_actions_by_street):
    return [
        [
            list(map(full_action_to_int, player_actions))
            for player_actions in street_actions
        ]
        for street_actions in full_actions_by_street
    ]

def player_actions_to_matrix(player_actions):
    action_matrix = np.zeros((4,4))
    for idx, actions in enumerate(player_actions):
        action_matrix[idx, actions] = 1
    return action_matrix