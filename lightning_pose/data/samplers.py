"""Samplers controlling how multi-dataset corpora are traversed during training.

The supervised heatmap loss weights every labeled keypoint equally, so a source
dataset's gradient influence is its share of labeled keypoint observations, not its
share of frames. :class:`TemperatureSampler` therefore defines its temperature over
supervision mass ``m_d = n_d * kbar_d`` (frames times mean labeled keypoints per
frame) and converts the target supervision share into a frame-draw probability:

    p_d ∝ m_d^(1/T)      target share of supervision
    q_d ∝ p_d / kbar_d   frame-draw probability

``T=1`` reduces to frame-proportional sampling (``q_d ∝ n_d``) — identical in
expectation to a plain shuffled traversal — so callers should not construct a
sampler at all for ``T=1`` and instead keep the stock ``shuffle=True`` loader;
this module is for ``T > 1`` up to ``T=inf`` (equal supervision per dataset).
"""

import logging
import math

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []

# frames drawn per multinomial call while materializing an epoch; only a
# throughput knob, has no effect on the sampled distribution
_DRAW_CHUNK = 4096


class TemperatureSampler(Sampler[int]):
    """Sample a dataset from ``q_d`` per draw, then the next frame from its shuffled queue.

    Epoch semantics (reset-on-exhaustion): the epoch ends the moment any dataset's
    queue is fully consumed, and every queue reshuffles for the next epoch. The epoch
    is therefore bounded by the smallest effective source, larger sources achieve
    statistical rather than guaranteed coverage within one epoch, and epoch length
    varies slightly epoch to epoch (the draw sequence is deterministic given
    ``(seed, epoch)``, but which draw exhausts the first queue is not fixed).
    Schedules must consequently be step-based, never epoch-based.

    Yields positions into the training subset it orders (0..n_train-1): the
    DataLoader that consumes this sampler wraps the train ``Subset``, and a sampler's
    values index into the dataset it is attached to — yielding full-dataset indices
    here would double-index through the Subset and silently select wrong frames.

    Args:
        dataset_ids: long tensor with one entry per training-subset position, giving
            each frame's source-dataset id.
        kbar: per-dataset mean labeled (``visible==2``) keypoints per frame,
            computed from the training subset only — using the full CSV would leak
            validation composition into the sampler.
        temperature: sampling temperature ``T > 1``; ``math.inf`` gives every
            dataset an equal supervision share. For ``T=1`` use the stock shuffled
            loader instead (this class refuses it to keep the equivalence exact).
        seed: base seed; epoch ``e`` draws with generator seed ``seed * 100_003 + e``
            so runs and restarts reproduce the same sequence.
    """

    def __init__(
        self,
        dataset_ids: torch.Tensor,
        kbar: torch.Tensor,
        temperature: float,
        seed: int = 42,
    ) -> None:
        if temperature <= 1:
            raise ValueError(
                f'temperature must be > 1 (got {temperature}); T=1 is frame-proportional '
                f'sampling — use the stock shuffle=True loader so the equivalence is exact'
            )

        self.dataset_ids = dataset_ids.long()
        self.num_frames = len(dataset_ids)
        self.temperature = float(temperature)
        self.seed = seed
        self._epoch = 0

        num_datasets = len(kbar)
        n_d = torch.bincount(self.dataset_ids, minlength=num_datasets).double()
        present = n_d > 0
        if not present.any():
            raise ValueError('training subset is empty')

        kbar = kbar.double()
        if (kbar[present] <= 0).any():
            raise ValueError(
                f'kbar must be positive for every present dataset, got {kbar.tolist()} — '
                f'a dataset whose training frames carry no visible=2 labels cannot be '
                f'assigned a supervision share'
            )

        # p_d ∝ m_d^(1/T) over present datasets; T=inf → uniform supervision share
        m_d = n_d * kbar
        p_d = torch.zeros_like(m_d)
        if math.isinf(self.temperature):
            p_d[present] = 1.0
        else:
            p_d[present] = m_d[present] ** (1.0 / self.temperature)
        p_d /= p_d.sum()

        # q_d ∝ p_d / kbar_d — the frame-draw probability that realizes p_d
        q_d = torch.zeros_like(p_d)
        q_d[present] = p_d[present] / kbar[present]
        q_d /= q_d.sum()

        self.n_d = n_d.long()
        self.p_d = p_d
        self.q_d = q_d

        # subset positions per dataset, fixed order; shuffled per epoch
        self._positions = [
            torch.nonzero(self.dataset_ids == d).flatten() for d in range(num_datasets)
        ]

        # expected draws until the scarcest queue empties, for logging/len before iteration
        with_frames = torch.nonzero(present).flatten()
        expected_len = min(
            float(n_d[d] / q_d[d]) for d in with_frames.tolist()
        )
        logger.info(
            f'TemperatureSampler T={self.temperature}: q_d={ [round(v, 4) for v in q_d.tolist()] }, '
            f'expected epoch length ~{expected_len:.0f} of {self.num_frames} train frames'
        )

        self._current: list[int] = self._materialize(self._epoch)

    def _materialize(self, epoch: int) -> list[int]:
        """Generate one full epoch's index sequence, deterministic given (seed, epoch)."""
        g = torch.Generator().manual_seed(self.seed * 100_003 + epoch)
        queues = [pos[torch.randperm(len(pos), generator=g)] for pos in self._positions]
        ptr = [0] * len(queues)
        out: list[int] = []
        while True:
            draws = torch.multinomial(self.q_d, _DRAW_CHUNK, replacement=True, generator=g)
            for d in draws.tolist():
                out.append(int(queues[d][ptr[d]]))
                ptr[d] += 1
                if ptr[d] == len(queues[d]):
                    # a queue just emptied: the epoch ends here, all queues reset next epoch
                    return out

    def __iter__(self):
        if self._current is None:
            self._current = self._materialize(self._epoch)
        current = self._current
        # advance: the next epoch draws from a different generator seed
        self._epoch += 1
        self._current = None
        return iter(current)

    def __len__(self) -> int:
        if self._current is None:
            self._current = self._materialize(self._epoch)
        return len(self._current)

    def state_dict(self) -> dict:
        """Serializable position in the epoch stream; restore() reproduces the sequence."""
        return {'seed': self.seed, 'epoch': self._epoch, 'temperature': self.temperature}

    def load_state_dict(self, state: dict) -> None:
        if state['seed'] != self.seed or state['temperature'] != self.temperature:
            raise ValueError(
                f'sampler state (seed={state["seed"]}, T={state["temperature"]}) does not '
                f'match this sampler (seed={self.seed}, T={self.temperature})'
            )
        self._epoch = state['epoch']
        self._current = None


class RepeatedEpochBatchSampler(Sampler[list[int]]):
    """Concatenate ``repeats`` independent shuffled passes over a subset into one epoch.

    Batch boundaries never cross a pass, so the batches are exactly those the stock
    ``shuffle=True, drop_last=False`` loader would produce over ``repeats`` consecutive
    epochs; only Lightning's per-epoch turnover (iterator reset, callbacks, progress bar,
    status writes) is amortized. Intended for few-frame training, where one pass is a
    single batch and that turnover otherwise dominates wall-clock. Step-based settings
    (``max_steps``, ``val_check_interval``, ``unfreezing_step``, ``milestone_steps``) are
    unaffected because the step count per pass is unchanged and ``milestone_steps`` is
    converted through ``len(train_dataloader)``, which this sampler reports correctly.

    Args:
        n: number of examples in the training subset
        batch_size: batch size of the stock loader being replaced
        repeats: number of shuffled passes per epoch
        generator: torch RNG that seeds every pass (one ``randperm`` call each)
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        repeats: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if n <= 0 or batch_size <= 0 or repeats <= 0:
            raise ValueError(
                f'n, batch_size and repeats must be positive, got {n}, {batch_size}, {repeats}'
            )
        self.n = n
        self.batch_size = batch_size
        self.repeats = repeats
        self.generator = generator if generator is not None else torch.Generator()

    def __iter__(self):
        for _ in range(self.repeats):
            perm = torch.randperm(self.n, generator=self.generator).tolist()
            for i in range(0, self.n, self.batch_size):
                yield perm[i:i + self.batch_size]

    def __len__(self) -> int:
        return self.repeats * math.ceil(self.n / self.batch_size)
