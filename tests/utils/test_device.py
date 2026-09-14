"""Test the device module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from lightning_pose.utils.device import get_accelerator, require_cuda_for_semi_supervised


class TestGetAccelerator:
    """Test the function get_accelerator."""

    def test_get_accelerator_returns_gpu_when_cuda_available(self):
        with patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=True):
            assert get_accelerator() == 'gpu'

    def test_get_accelerator_returns_cpu_when_cuda_unavailable(self):
        with patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=False):
            assert get_accelerator() == 'cpu'


class TestRequireCudaForSemiSupervised:
    """Test the function require_cuda_for_semi_supervised.

    Mocks at the same boundaries as tests/data/video/test_factory.py: CUDA availability via
    ``lightning_pose.utils.device.torch.cuda.is_available``, and dali's importability via
    ``sys.modules`` substitution (``None`` forces the ImportError a real missing install
    would raise; a ``MagicMock`` stands in for a successful import).
    """

    def test_noop_when_losses_to_use_is_empty(self):
        with patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=False):
            require_cuda_for_semi_supervised([])

    def test_noop_when_losses_to_use_is_none(self):
        with patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=False):
            require_cuda_for_semi_supervised(None)

    def test_raises_without_cuda(self):
        with patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match='requires an NVIDIA GPU with CUDA'):
                require_cuda_for_semi_supervised(['pca_singleview'])

    def test_raises_without_dali(self):
        with (
            patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=True),
            patch.dict(sys.modules, {'lightning_pose.data.video.dali': None}),
        ):
            with pytest.raises(RuntimeError, match='requires the nvidia-dali package'):
                require_cuda_for_semi_supervised(['pca_singleview'])

    def test_passes_with_cuda_and_dali(self):
        fake_dali_module = MagicMock()
        with (
            patch('lightning_pose.utils.device.torch.cuda.is_available', return_value=True),
            patch.dict(sys.modules, {'lightning_pose.data.video.dali': fake_dali_module}),
        ):
            require_cuda_for_semi_supervised(['pca_singleview'])
