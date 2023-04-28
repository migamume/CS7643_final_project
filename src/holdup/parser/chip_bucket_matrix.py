from typing import List, Tuple

import numpy as np

from holdup.parser.player_actions import CALL, RAISE

# Chip counts associated with actions
# 8x20, subdivided buckets

# 19 Bins because we need the 0 bucket to have special behavior
# ie: We need Bin 1 to be 0 and Bin 2 to be 1
num_bins=19
"""
Note: Some interesting experiments can be had if we decide to bucket differently
We could also visualize the distribution of sizes across the dataset to inform choice
May need to normalize across datasets for valid comparison
Todo: Make configurable
"""
max_value= 20_000 # Verified max val is 20k                                                         
bins = np.linspace(1, max_value, num_bins)

def action_to_bucket(full_action: Tuple[str, int]) -> int:
    action_name = full_action[0]
    if action_name == CALL or action_name == RAISE:
        # Normally we would subtract one here to make the bin 0 indexed
        # but, as per comment above, we want these bins 1 indexed
        return np.digitize(full_action[1], bins)
    else:
        return 0

def street_chip_buckets(full_actions_by_street: List[List[List[Tuple[str, int]]]]) -> List[List[List[int]]]:
    return [
        [
            list(map(action_to_bucket, player_actions))
            for player_actions in street_actions
        ]
        for street_actions in full_actions_by_street
    ]

def chip_buckets_to_matrix(chip_buckets: List[List[List[int]]]) -> np.ndarray:
    chip_matrix = np.zeros((8, 20))
    for street_idx, buckets in enumerate(chip_buckets):
        start_idx = street_idx * 2
        end_idx = start_idx + 1
        chip_matrix[start_idx, buckets[0]] = 1
        chip_matrix[end_idx, buckets[1]] = 1
    return chip_matrix