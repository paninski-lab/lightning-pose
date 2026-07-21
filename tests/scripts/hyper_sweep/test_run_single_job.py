"""End-to-end test for scripts/hyper-sweep/run_single_job.py.

Downloads the real, ~1MB `paninski-lab/mirror-mouse-tiny` test-fixture dataset from
HuggingFace and runs a tiny (`--debug`) training job through it, checking that the whole
pipeline -- dataset download/extraction, `litpose train`, and checkpoint cleanup -- still
works end to end. Hits the real HuggingFace repo on every run; no mocking.
"""

import sys


class TestMain:
    """Test the main function against the real paninski-lab/mirror-mouse-tiny dataset."""

    def test_main_runs_debug_job_without_error(self, run_single_job, tmp_path, monkeypatch):
        output_dir = tmp_path / 'output'
        monkeypatch.setattr(sys, 'argv', [
            'run_single_job.py',
            '--dataset_repo=paninski-lab/mirror-mouse-tiny',
            f'--local_data_dir={tmp_path / "data"}',
            '--backbone=resnet18',
            '--train_frames=1',
            '--seed=0',
            f'--output_dir={output_dir}',
            '--debug',  # runs for 3 epochs
        ])

        run_single_job.main()

        # trained to completion and produced predictions
        assert (output_dir / 'predictions.csv').exists()
        # checkpoints are deleted to save storage
        assert not list(output_dir.rglob('*.ckpt'))
