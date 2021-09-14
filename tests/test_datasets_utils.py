from torch.utils.data import random_split
from pose_est_nets.datasets.utils import split_sizes_from_probabilities


def test_split_sizes_from_probabilities():
    """assuming a known datalength = 103, and probs 0.8, 0.1, 0.1,
    make sure we count examples properly"""
    total_number = 103
    out = random_split(
        range(total_number), split_sizes_from_probabilities(total_number, 0.8, 0.1, 0.1)
    )
    assert len(out[0]) == 82 and len(out[1]) == 10 and len(out[2]) == 11
    out = random_split(
        range(total_number), split_sizes_from_probabilities(total_number, 0.8, 0.1)
    )  # letting it compute test_number
    assert len(out[0]) == 82 and len(out[1]) == 10 and len(out[2]) == 11
