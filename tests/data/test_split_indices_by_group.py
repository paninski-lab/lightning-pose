"""Tests for lightning_pose.data.utils.split_indices_by_group."""

import pytest
import torch

from lightning_pose.data.utils import (
    split_indices_by_group,
    split_sizes_from_probabilities,
)


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


def _paired_group_ids(n_frames, n_animals=2):
    """Row layout of a per-animal dataset: frame-major, animal-minor."""
    return [f"frame{i}" for i in range(n_frames) for _ in range(n_animals)]


def _split_of(splits):
    """row index -> which split it landed in."""
    return {row: i for i, rows in enumerate(splits) for row in rows}


def test_animals_of_a_frame_never_land_in_different_splits():
    group_ids = _paired_group_ids(100)
    splits = split_indices_by_group(group_ids, [190, 10, 0], _generator())
    where = _split_of(splits)

    by_group = {}
    for row, gid in enumerate(group_ids):
        by_group.setdefault(gid, set()).add(where[row])

    assert all(len(s) == 1 for s in by_group.values())


def test_every_row_is_used_exactly_once():
    group_ids = _paired_group_ids(50)
    splits = split_indices_by_group(group_ids, [90, 5, 5], _generator())
    assert sorted(r for rows in splits for r in rows) == list(range(len(group_ids)))


def test_rows_of_a_group_are_contiguous():
    """train_frames truncates on a group boundary, so grouping must be preserved."""
    group_ids = _paired_group_ids(40)
    train, _, _ = split_indices_by_group(group_ids, [60, 10, 10], _generator())

    order = []
    for row in train:
        gid = group_ids[row]
        if gid not in order:
            order.append(gid)
        else:
            assert order[-1] == gid, "group rows are not contiguous"


def test_split_is_reproducible_for_a_seed():
    group_ids = _paired_group_ids(30)
    assert (
        split_indices_by_group(group_ids, [50, 5, 5], _generator(7))
        == split_indices_by_group(group_ids, [50, 5, 5], _generator(7))
    )
    assert (
        split_indices_by_group(group_ids, [50, 5, 5], _generator(7))
        != split_indices_by_group(group_ids, [50, 5, 5], _generator(8))
    )


def test_realised_sizes_are_close_to_requested():
    # Groups are indivisible, so sizes can only be hit to within one group.
    group_ids = _paired_group_ids(100)
    train, val, test = split_indices_by_group(group_ids, [160, 20, 20], _generator())

    assert abs(len(train) - 160) <= 2
    assert abs(len(val) - 20) <= 2
    assert len(train) + len(val) + len(test) == 200


def test_zero_target_split_stays_empty():
    """A request for no test set must not gain one from group granularity."""
    group_ids = _paired_group_ids(50)
    train, val, test = split_indices_by_group(group_ids, [95, 5, 0], _generator())

    assert test == []
    assert len(train) + len(val) == 100
    assert len(train) % 2 == 0 and len(val) % 2 == 0  # whole groups only


def test_handles_variable_group_sizes():
    # An animal missing from some frames gives groups of different sizes.
    group_ids = ["a", "a", "b", "c", "c", "c", "d", "e", "e"]
    splits = split_indices_by_group(group_ids, [5, 2, 2], _generator())
    where = _split_of(splits)

    by_group = {}
    for row, gid in enumerate(group_ids):
        by_group.setdefault(gid, set()).add(where[row])
    assert all(len(s) == 1 for s in by_group.values())
    assert sum(len(s) for s in splits) == len(group_ids)


def test_singleton_groups_match_an_ungrouped_split_size():
    group_ids = [f"frame{i}" for i in range(100)]
    sizes = split_sizes_from_probabilities(100, 0.8, 0.1, 0.1)
    assert [len(s) for s in split_indices_by_group(group_ids, sizes, _generator())] == sizes


def test_raises_on_empty_input():
    with pytest.raises(ValueError, match="empty"):
        split_indices_by_group([], [1, 1, 0], _generator())


def test_raises_when_validation_would_be_empty():
    # Two groups of 5 cannot fill a 9-row train split and still leave a val row.
    with pytest.raises(ValueError, match="empty validation set"):
        split_indices_by_group(["a"] * 5 + ["b"] * 5, [9, 1, 0], _generator())


def test_all_zero_targets_raises():
    with pytest.raises(ValueError, match="All split sizes are zero"):
        split_indices_by_group(["a", "a"], [0, 0, 0], _generator())
