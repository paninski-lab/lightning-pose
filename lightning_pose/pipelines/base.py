"""Abstract base class and result type shared by all inference pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lightning_pose.utils.inference_types import _Reader

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


@dataclass
class PipelineResult:
    """Output of a pipeline run on one video.

    Attributes:
        predictions: standard LP prediction DataFrame, one row per frame, with keypoint
            coordinates in the original (full-resolution) frame space.
        bboxes: standard LP bbox DataFrame (columns ``x, y, h, w``), one row per frame, in
            original frame coordinates.
    """

    predictions: pd.DataFrame
    bboxes: pd.DataFrame


class Pipeline(ABC):
    """Base class for inference pipelines that chain several trained models.

    Subclasses own the data flow between their models (what is passed from one stage to the
    next, and when intermediates are freed). They are constructed from already-built
    ``Model`` objects, so they never touch checkpoints or training.
    """

    @abstractmethod
    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: str | Path,
        reader: _Reader | None = None,
        progress_file: Path | None = None,
    ) -> PipelineResult:
        """Run the full pipeline on a video and save its outputs to ``output_dir``.

        Args:
            video_file: path to the video file.
            output_dir: directory to write the prediction CSV (and any intermediate CSVs,
                e.g. bboxes) to.
            reader: video-reading backend; None auto-selects, as in
                ``Model.predict_on_video_file``.
            progress_file: path to a file used to report progress to the App.

        Returns:
            PipelineResult with the predictions and intermediates that were saved.
        """
