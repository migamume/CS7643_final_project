import bz2
from io import BytesIO
import numpy as np
import os
import re
import shutil
import tarfile
import urllib.request

from holdup.parser.action_matrix import player_actions_to_matrix, street_int_actions
from holdup.parser.chip_bucket_matrix import street_chip_buckets, chip_buckets_to_matrix
from holdup.parser.hand_matrix import make_hand_matrix
from holdup.parser.player_actions import raw_hand_to_tuple
from holdup.utils import LogProvider

logger = LogProvider.get_logger()


# Note: I did see a hand history file that slightly changed this
# It had the players first and then STATE:...
# If we need to use that year's data, it's an easy fix to define a fallback regex
proposed_regex = "STATE:\d:(\w*\/?\w*\/?\w*\/?\w*):(\w{4})\|(\w{4})(\/?\w*\/?\w*\/?\w*)?:(-?\d*)\|(-?\d*):([^|]*)\|(.*)"

urls = [
    "http://www.computerpokercompetition.org/downloads/competitions/2011/logs/acpc_2011_2p_nolimit.tar.bz2"
    # Add more URLs as needed
]

"""
Some files have headers that we want to ignore
This is a convenience function to let us filter them out
"""
def regex_matches(raw_hand):
    matches = re.findall(proposed_regex, raw_hand)

    if len(matches) == 0:
        return False
    else: return True


"""
Converts a raw hand matched by the regex
    (ex: "STATE:4:r300c/cr600f:8dKd|2cKc/2d5c7s:-300|300:hugh|Hyperborean-2011-2p-nolimit-iro")
into a matrix defined by the paper.
To do so, parsed to an easier to use tupled form
Then use functions to produce each matrix component A, B, C
Then stack them
"""
def make_matrix(raw_hand: str):
    # Todo: Account for diff regex orders; maybe need to have a fallback
    matches = re.findall(proposed_regex, raw_hand)

    action, player_a_cards, player_b_cards, community_cards, _, _, player_a, player_b = matches[0]
    tupled_hand = raw_hand_to_tuple(action, player_a, player_b)

    street_full_actions = list(map(lambda street: [street[1], street[2]], tupled_hand))

    _street_int_actions = street_int_actions(street_full_actions)
    player_a_actions = [round_actions[0] for round_actions in _street_int_actions]
    player_b_actions = [round_actions[1] for round_actions in _street_int_actions]

    both_player_action_matrix = np.vstack(
        (
            player_actions_to_matrix(player_a_actions),
            player_actions_to_matrix(player_b_actions)
        )
    )

    player_cards = np.vstack((
        make_hand_matrix(community_cards, player_a_cards),
        make_hand_matrix(community_cards, player_b_cards)
    ))

    scb = street_chip_buckets(street_full_actions)
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
    return a_b_c


def map_to_matricies(file_contents):
    # File contents are newline delimited unsparsed string representations of hands
    return [
        make_matrix(line).flatten()
        for line in file_contents.split("\n")
        if line != "" and regex_matches(line)
    ]


def download_and_extract(out_dir: str, url: str):
    logger.info(f"Downloading file at {url}. This could take a while")
    with urllib.request.urlopen(url) as response:
        compressed_data = response.read()

    logger.info("File downloaded. Uncompressing bz2")
    decompressed_data = bz2.decompress(compressed_data)

    logger.info("Decompressed. Extracting from tar")
    with tarfile.open(fileobj=BytesIO(decompressed_data), mode="r") as tar:
        tar.extractall(out_dir)


"""
Download hands from a URL, make "raw" and "parsed" directories,
and parsed and move files appropriately.
"""
def download_parse_persist(url: str):
    # ie: The file name without an extension or parent dir info
    file_prefix = os.path.splitext(os.path.splitext(os.path.basename(url))[0])[0]
    # Note: Won't work if we run in a zip. But... we won't
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_out_dir = os.path.join(current_dir, 'data', file_prefix)

    if not os.path.exists(base_out_dir):
        logger.info("Creating %s" % base_out_dir)
        download_and_extract(base_out_dir, url)

        raw_dir = os.path.join(base_out_dir, "raw")
        parsed_dir = os.path.join(base_out_dir, "parsed")
        for dir in [raw_dir, parsed_dir]: os.mkdir(dir)

        # Extracted files are nested in a non-deterministic # of sub dirs
        # Bring them all up to root for easier usage
        for root, _, files in os.walk(base_out_dir):
            for file in files:
                # NOTE: This might not be consistent across years.
                # Another fallback maybe needed here
                if file.endswith(".log"):
                    raw_file = os.path.join(root, file)
                    # Note: I am moving the files but not cleaning the dirs
                    shutil.move(raw_file, os.path.join(raw_dir, file))

        for filename in os.listdir(raw_dir):
            with open(os.path.join(raw_dir, filename), 'r') as raw_file:
                parsed = map_to_matricies(raw_file.read())
                np.savetxt(parsed, os.path.join(parsed_dir, f'{filename}_parsed.csv'), delimiter=",")

if __name__ == "__main__":
    download_parse_persist(urls[0])
