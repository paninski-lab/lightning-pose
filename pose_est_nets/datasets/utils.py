from typing import Optional, List
import numpy as np
from typeguard import typechecked


@typechecked
def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: float,
    test_probability: Optional[float] = None,
) -> List[int]:
    """a utility that takes in dataset length, and some probabilities for slicing, and spits out the number of examples for training, validation, testing.

    Args:
        total_number (int): total number of examples in dataset
        train_probability (float): fraction of examples used for training
        val_probability (float): fraction of examples used for validation
        test_probability (Optional[float], optional): fraction of examples used for test. Defaults to None. Can be computed as the remaining examples.

    Returns:
        List[int]: num training examples, num validation examples, num test examples
    """

    if test_probability is None:
        test_probability = 1.0 - train_probability - val_probability
    assert (
        test_probability + train_probability + val_probability == 1.0
    )  # probabilities should add to one
    train_number = int(np.floor(train_probability * total_number))
    val_number = int(np.floor(val_probability * total_number))
    test_number = (
        total_number - train_number - val_number
    )  # if we lose extra examples by flooring, send these to test_number
    assert (
        train_number + test_number + val_number == total_number
    )  # assert that we're using all datapoints
    return [train_number, val_number, test_number]
