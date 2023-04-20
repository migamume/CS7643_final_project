from holdup.the_model.autoencoder import Autoencoder
import numpy as np
# import pandas as pd
import torch
from torch import nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
import random
import os
from holdup.parser.replayable_hand import ReplayableHand, Streets
import functools
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

#from David's updated_parser.ipynb
"""
- Last possible action to final action
tuple(pf, f, t, r)
preflop: List[Matrix, Integer: Supervised Target]
flop: List[Matrix, Integer: Supervised Target]
turn: List[Matrix, Integer: Supervised Target]
river: List[Matrix, Integer: Supervised Target]

- Every possible action to subsequent action; not across round boundaries
preflop: List[Matrix, Integer: Supervised Target]
flop: List[Matrix, Integer: Supervised Target]
turn: List[Matrix, Integer: Supervised Target]
river: List[Matrix, Integer: Supervised Target]
"""

def reducer(reduce_type: str):
    def doit(agg: Tuple[
        List[Tuple[np.ndarray, Tuple[bool, int]]],
        List[Tuple[np.ndarray, Tuple[bool, int]]],
        List[Tuple[np.ndarray, Tuple[bool, int]]],
        List[Tuple[np.ndarray, Tuple[bool, int]]]
    ], hands_dict):

        if reduce_type == "last_possible":
            # Get the 2nd to last action
            appender = lambda street, actions: street + [actions[-2]] if len(actions) > 1 else street
        else:
            # Get all actions up to the last action
            appender = lambda street, actions: street + actions[0:-1] if len(actions) > 1 else street

        return [
            appender(actions_for_training, hands_dict[street_name])
            for street_name, actions_for_training in zip(
                (Streets.Preflop, Streets.Flop, Streets.Turn, Streets.River),
                agg
            )
        ]

    return doit



#modified from David's updated parser notebook
np.random.seed(42)
def get_datasets(last_possible = True):
    data_dir = r'holdup/parser/data/acpc_2011_2p_nolimit/raw'
    log_files = [data_dir + "/" + f for f in os.listdir(data_dir) if f.endswith('.log')]
    random.shuffle(log_files)

    # Process each log file and reduce it with the 'last_possible' reducer
    if last_possible:
        results = [
            functools.reduce(
                reducer('last_possible'),
                [
                    # Dict[int, List[Tuple[np.ndarray, Tuple[bool, int]]]]
                    ReplayableHand(line.strip()).matrix_next_action_by_street()
                    for line in open(log_file, 'r')
                    if line.strip() != ''
                ],
                ([], [], [], [])
            )
            for log_file in log_files
        ]
        # output_file_prefix = 'last_possible'
    else:
        results = [
            functools.reduce(
                reducer('lmao'),
                [
                    ReplayableHand(line.strip()).matrix_next_action_by_street()
                    for line in open(log_file, 'r')
                    if line.strip() != ''
                ],
                ([], [], [], [])
            )
            for log_file in log_files
        ]

    # (pre-flop, flop, turn, river)
    return results