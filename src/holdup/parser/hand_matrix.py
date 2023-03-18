import numpy as np

# TODO
rank_mappings = {'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4, '9': 5, '8': 6, '7': 7, '6': 8, '5': 9, '4': 10, '3': 11, '2': 12}
suit_mappings = {'c': 0, 'd': 1, 'h': 2, 's': 3}

def vector_location(hand):
    return np.array([suit_mappings[hand[1]], rank_mappings[hand[0]]])

def make_hand_matrix(community_cards, player_cards):
    deck = np.zeros((4,13))
    # Turn slash separated string of streets into a single string of cards
    # Then join all cards and stride with len 2 to stride over one card at a time
    ccs = community_cards.split('/')[1:] # Throw away empty first element
    all_cards = "".join(ccs) + player_cards
    card_locations = [
        vector_location(all_cards[i:i+2])
        for i in range(0, len(all_cards), 2)
    ]
    # Is there a faster vectorized way? Probably doesn't matter atm
    for card_location in card_locations:
        deck[card_location[0], card_location[1]] = 1
    return deck