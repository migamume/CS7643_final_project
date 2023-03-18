import functools
import re


CALL = "CALL"
CHECK = "CHECK"
FOLD = "FOLD"
RAISE = "RAISE"

action_mappings = {"c": ["CHECK", "CALL"], "f": "FOLD", "r": "RAISE"}

row_regex = "STATE:\d:(\w*\/?\w*\/?\w*\/?\w*):(\w{4})\|(\w{4})(\/?\w*\/?\w*\/?\w*)?:(-?\d*)\|(-?\d*):([^|]*)\|(.*)"

def raw_hand_to_tuple(actions, player_a, player_b):
    by_round = actions.split("/")

    def reducer(agg, element):
        prev_token, player_a_actions, player_b_actions, chips, current_player = agg
        if len(prev_token) == 0:
            action = action_mappings[element]
            if isinstance(action, list):
                return CHECK, [(CHECK, 0)] + player_a_actions, player_b_actions, chips, player_b
            if action == FOLD:
                return FOLD, [(FOLD, 0)] + player_a_actions, player_b_actions, chips, player_b
            # We need to build up the chips in the raise action over multiple iterations
            return RAISE, player_a_actions, player_b_actions, chips, player_a

        # We have a chip amount instead of an action. ie: still collecting the total raise/call size
        if element not in action_mappings.keys():
            return prev_token, player_a_actions, player_b_actions, chips + element, current_player

        # We have a new action for a new player
        action = action_mappings[element]

        if prev_token == CHECK:
            if isinstance(action, list):
                # Can only go check-check when player a+b each take 1 action
                return CHECK, player_a_actions, [(CHECK, 0)] + player_b_actions, chips, player_a
            if action == FOLD:
                raise Exception("A bot should never fold when checked to!")
            return RAISE, player_a_actions, player_b_actions, chips, current_player

        if prev_token == RAISE:
            # Previously collected chips need to be converted into a proper player action now
            num_chips = int(chips)
            full_chip_based_action = [(prev_token, num_chips)]
            new_chip_tracker = ""

            # Can only call after raise
            if isinstance(action, list):
                # Note: Same player stays current_player because we are recording two actions
                # 1: The accumlated raise, 2: The response
                if current_player == player_a:
                    return CALL, full_chip_based_action + player_a_actions, [(CALL, num_chips)] + player_b_actions, new_chip_tracker, player_a
                else:
                    return CALL, [(CALL, num_chips)] + player_a_actions, full_chip_based_action + player_b_actions, new_chip_tracker, player_b

            if action == FOLD:
                full_action = [(action, 0)]
                if current_player == player_a:
                    return action, full_chip_based_action + player_a_actions, full_action + player_b_actions, new_chip_tracker, player_a
                else:
                    return action, full_action + player_a_actions, full_chip_based_action + player_b_actions, new_chip_tracker, player_a

            if action == RAISE:
                if current_player == player_a:
                    return action, full_chip_based_action + player_a_actions, player_b_actions, new_chip_tracker, player_b
                else:
                    return action, player_a_actions, full_chip_based_action + player_b_actions, new_chip_tracker, player_a

        # prev_token can't be equal to call because iteration would already have stopped
        raise Exception("This code should be unreachable")

    rounds = [functools.reduce(reducer, _round, ["", [], [], "", player_a]) for _round in by_round]
    # Todo: Reverse the action lists for each player
    #   Note: Actually, order doesn't matter if we're just stamping in the matrix.
    # Todo: Should also export who player A and B is as columns. They drop them before training
    #   This allows us to potentially train player based models later with no need to refactor this code
    return rounds

def demo_hands():
    hands = """STATE:0:r241f:8h4c|6c5h:-100|100:act1_2pn_2016|hugh_2pn_2016
        STATE:1:r223f:6h9c|Jc8h:-100|100:hugh_2pn_2016|act1_2pn_2016
        STATE:2:r241r675c/cr1629c/cc/r2606c:QcAc|TsAs/JsTh7d/Qh/Td:-2606|2606:act1_2pn_2016|hugh_2pn_2016
        STATE:3:cc/r241c/r581f:2dQh|6d3s/Jc4h2c/Tc:241|-241:hugh_2pn_2016|act1_2pn_2016
        STATE:4:r241r809c/cr1953c/cr4714f:8cTc|9sAd/3cAc3s/Js:-1953|1953:act1_2pn_2016|hugh_2pn_2016
        STATE:5:r273f:9d4c|JdTd:-100|100:hugh_2pn_2016|act1_2pn_2016
        STATE:6:r241r675c/cr1629c/cc/cc:9c9s|7s7d/AcAhKh/5c/Qc:1629|-1629:act1_2pn_2016|hugh_2pn_2016
        STATE:7:r273f:4d3d|KdQh:-100|100:hugh_2pn_2016|act1_2pn_2016
        STATE:8:r241f:7c5s|6h6s:-100|100:act1_2pn_2016|hugh_2pn_2016"""

    matches = list(map(lambda hand: re.findall(row_regex, hand), hands.split("\n")))
    return [raw_hand_to_tuple(row[0][0], row[0][-2], row[0][-1]) for row in matches]

if __name__ == '__main__':
    print(demo_hands())