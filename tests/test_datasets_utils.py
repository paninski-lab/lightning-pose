from pose_est_nets.datasets.utils import split_sizes_from_probabilities


def test_split_sizes_from_probabilities():
    """make sure we count examples properly"""

    total_number = 100
    train_prob = 0.8
    val_prob = 0.1
    test_prob = 0.1

    out = split_sizes_from_probabilities(
        total_number,
        train_probability=train_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number,
        train_probability=train_prob,
        val_probability=val_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number,
        train_probability=train_prob,
        val_probability=val_prob,
        test_probability=test_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(total_number, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 15

    # test that extra samples end up in test
    out = split_sizes_from_probabilities(101, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 16
