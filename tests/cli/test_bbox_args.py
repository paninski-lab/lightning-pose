"""Test the bbox sizing arguments shared by create_bbox and predict."""

import argparse

import pytest

from lightning_pose.cli.bbox_args import add_bbox_sizing_args, resolve_bbox_sizing


class TestAddBboxSizingArgs:
    """Test the function add_bbox_sizing_args."""

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_bbox_sizing_args(parser)
        args = parser.parse_args([])
        assert args.crop_ratio is None
        assert args.crop_size is None
        assert args.anchor_keypoints == ''

    def test_parses_values(self):
        parser = argparse.ArgumentParser()
        add_bbox_sizing_args(parser)
        args = parser.parse_args(['--crop_size', '200', '--anchor_keypoints', 'a,b'])
        assert args.crop_size == 200
        assert args.anchor_keypoints == 'a,b'

    def test_works_on_argument_group(self):
        parser = argparse.ArgumentParser()
        add_bbox_sizing_args(parser.add_argument_group('group'))
        assert parser.parse_args(['--crop_ratio', '1.5']).crop_ratio == 1.5


class TestResolveBboxSizing:
    """Test the function resolve_bbox_sizing."""

    def test_default_is_crop_ratio_two(self):
        sizing, anchors = resolve_bbox_sizing(None, None, '')
        assert sizing == {'crop_ratio': 2.0}
        assert anchors == []

    def test_crop_ratio(self):
        sizing, _ = resolve_bbox_sizing(3.0, None, '')
        assert sizing == {'crop_ratio': 3.0}

    def test_crop_size_sets_height_and_width(self):
        sizing, _ = resolve_bbox_sizing(None, 128, '')
        assert sizing == {'crop_height': 128, 'crop_width': 128}

    def test_anchor_keypoints_split(self):
        _, anchors = resolve_bbox_sizing(None, None, 'nose,tail')
        assert anchors == ['nose', 'tail']

    def test_both_sizes_raise(self):
        with pytest.raises(ValueError, match='mutually exclusive'):
            resolve_bbox_sizing(2.0, 100, '')

    def test_non_positive_crop_size_raises(self):
        with pytest.raises(ValueError, match='positive integer'):
            resolve_bbox_sizing(None, 0, '')

    @pytest.mark.parametrize('crop_ratio', [1.0, 0.5])
    def test_crop_ratio_not_above_one_raises(self, crop_ratio):
        with pytest.raises(ValueError, match='greater than 1'):
            resolve_bbox_sizing(crop_ratio, None, '')
