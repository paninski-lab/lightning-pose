"""Test the models package-level functions."""

from lightning_pose.models import check_if_semi_supervised


def test_check_if_semisupervised():
    flag = check_if_semi_supervised(losses_to_use=None)
    assert not flag

    flag = check_if_semi_supervised(losses_to_use=[])
    assert not flag

    flag = check_if_semi_supervised(losses_to_use=[""])
    assert not flag

    flag = check_if_semi_supervised(losses_to_use=["any_string"])
    assert flag

    flag = check_if_semi_supervised(losses_to_use=["loss1", "loss2"])
    assert flag
