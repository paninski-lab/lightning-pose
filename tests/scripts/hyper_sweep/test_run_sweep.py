"""Test the scripts/hyper-sweep/run_sweep.py helper functions."""


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_reads_yaml_file(self, run_sweep, tmp_path):
        """Round-trips a minimal sweep config through yaml.safe_load."""
        config_path = tmp_path / 'sweep_config.yaml'
        config_path.write_text('sweep:\n  datasets:\n    - paninski-lab/mirror-mouse-fused\n')

        cfg = run_sweep.load_config(str(config_path))

        assert cfg == {'sweep': {'datasets': ['paninski-lab/mirror-mouse-fused']}}


class TestLossesStr:
    """Test the losses_str function."""

    def test_losses_str_empty_tuple_is_supervised(self, run_sweep):
        assert run_sweep.losses_str(()) == 'supervised'

    def test_losses_str_single_loss(self, run_sweep):
        assert run_sweep.losses_str(('temporal',)) == 'temporal'

    def test_losses_str_sorts_multiple_losses(self, run_sweep):
        assert run_sweep.losses_str(('temporal', 'pca_singleview')) == 'pca_singleview+temporal'


class TestSanitize:
    """Test the sanitize function."""

    def test_sanitize_replaces_slashes_and_dots(self, run_sweep):
        assert run_sweep.sanitize('resnet50.animal/ap10k') == 'resnet50_animal_ap10k'

    def test_sanitize_casts_non_string_input(self, run_sweep):
        assert run_sweep.sanitize(1) == '1'


class TestDatasetShortname:
    """Test the dataset_shortname function."""

    def test_dataset_shortname_strips_org(self, run_sweep):
        result = run_sweep.dataset_shortname('paninski-lab/mirror-mouse-fused')
        assert result == 'mirror-mouse-fused'


class TestMakeJobName:
    """Test the make_job_name function."""

    def test_make_job_name_joins_all_fields(self, run_sweep):
        name = run_sweep.make_job_name(
            'paninski-lab/mirror-mouse-fused', 'resnet50_animal_ap10k', 1, 0, ('temporal',),
        )
        assert name == 'mirror-mouse-fused__resnet50_animal_ap10k__temporal__tf1__s0'

    def test_make_job_name_supervised_only(self, run_sweep):
        name = run_sweep.make_job_name(
            'paninski-lab/mirror-mouse-fused', 'resnet50_animal_ap10k', 1, 0, (),
        )
        assert name == 'mirror-mouse-fused__resnet50_animal_ap10k__supervised__tf1__s0'


class TestMakeOutputDir:
    """Test the make_output_dir function."""

    def test_make_output_dir_builds_nested_path(self, run_sweep):
        out_dir = run_sweep.make_output_dir(
            '/base', 'paninski-lab/mirror-mouse-fused', 'resnet50_animal_ap10k', 1, 0,
            ('temporal',),
        )
        assert out_dir == '/base/mirror-mouse-fused/resnet50_animal_ap10k/temporal/tf1/seed0'


class TestMakeWorkerCommand:
    """Test the make_worker_command function."""

    def test_make_worker_command_includes_all_overrides(self, run_sweep):
        combo = ('paninski-lab/mirror-mouse-fused', 'resnet50_animal_ap10k', 1, 0, ('temporal',))
        cfg = {
            'output': {'base_dir': '/base', 'local_data_dir': '/tmp/lp_data'},
            'sweep': {'predict_vids_after_training': True, 'model_type': 'heatmap'},
            'debug': True,
        }

        cmd = run_sweep.make_worker_command(combo, cfg, 'run_single_job.py')

        assert '--dataset_repo=paninski-lab/mirror-mouse-fused' in cmd
        assert '--backbone=resnet50_animal_ap10k' in cmd
        assert '--losses_to_use=temporal' in cmd
        assert '--model_type=heatmap' in cmd
        out_dir = '/base/mirror-mouse-fused/resnet50_animal_ap10k/temporal/tf1/seed0'
        assert f'--output_dir={out_dir}' in cmd
        assert '--predict_vids' in cmd
        assert '--debug' in cmd

    def test_make_worker_command_omits_optional_flags_by_default(self, run_sweep):
        combo = ('paninski-lab/mirror-mouse-fused', 'resnet50_animal_ap10k', 1, 0, ())
        cfg = {'output': {'base_dir': '/base'}, 'sweep': {}}

        cmd = run_sweep.make_worker_command(combo, cfg, 'run_single_job.py')

        assert '--predict_vids' not in cmd
        assert '--debug' not in cmd
