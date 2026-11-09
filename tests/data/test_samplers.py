"""Tests for the multi-dataset temperature sampler."""

import math

import pytest
import torch

from lightning_pose.data.samplers import TemperatureSampler


def _ids(counts: list[int]) -> torch.Tensor:
    return torch.cat([torch.full((n,), d, dtype=torch.long) for d, n in enumerate(counts)])


class TestTemperatureSampler:
    """Test the class TemperatureSampler."""

    def test_temperature_sampler_refuses_t1(self):
        with pytest.raises(ValueError, match='must be > 1'):
            TemperatureSampler(_ids([10, 5]), torch.tensor([2.0, 2.0]), temperature=1.0)

    def test_temperature_sampler_inf_gives_equal_supervision(self):
        sampler = TemperatureSampler(_ids([100, 10]), torch.tensor([4.0, 1.0]), math.inf)
        assert torch.allclose(sampler.p_d, torch.tensor([0.5, 0.5], dtype=torch.double))
        # q_d ∝ p_d / kbar_d: the dataset with 4x the keypoints per frame is drawn 4x less
        assert sampler.q_d[1] / sampler.q_d[0] == pytest.approx(4.0)

    def test_temperature_sampler_share_follows_mass(self):
        sampler = TemperatureSampler(_ids([400, 100]), torch.tensor([1.0, 1.0]), temperature=2.0)
        # m = (400, 100) -> p ∝ (20, 10)
        assert sampler.p_d[0] / sampler.p_d[1] == pytest.approx(2.0)

    def test_temperature_sampler_frames_override(self):
        override = torch.tensor([float('nan'), 400.0])
        sampler = TemperatureSampler(
            _ids([400, 100]), torch.tensor([1.0, 1.0]), temperature=2.0, frames_override=override,
        )
        assert sampler.p_d[0] == pytest.approx(sampler.p_d[1])
        assert sampler.n_d.tolist() == [400, 100]

    def test_temperature_sampler_deterministic_valid_positions(self):
        ids = _ids([30, 20, 10])
        kbar = torch.tensor([1.0, 2.0, 3.0])
        a = list(TemperatureSampler(ids, kbar, temperature=3.0, seed=7))
        b = list(TemperatureSampler(ids, kbar, temperature=3.0, seed=7))
        assert a == b
        assert all(0 <= i < len(ids) for i in a)
        assert len(set(a)) == len(a)  # no frame twice within one epoch

    def test_temperature_sampler_state_dict_roundtrip(self):
        ids = _ids([30, 20])
        kbar = torch.tensor([1.0, 1.0])
        sampler = TemperatureSampler(ids, kbar, temperature=2.0, seed=3)
        list(sampler)
        second = list(sampler)
        restored = TemperatureSampler(ids, kbar, temperature=2.0, seed=3)
        restored.load_state_dict({'seed': 3, 'epoch': 1, 'temperature': 2.0})
        assert list(restored) == second
