"""Tests for bbox coordinate transformation utilities."""
import numpy as np
import pytest
import torch

from lightning_pose.data.bboxes import (
    convert_bbox_coords,
    convert_original_to_model_coords,
    normalized_to_bbox,
    original_to_model,
)


class TestNormalizedToBbox:
    """Test the function normalized_to_bbox."""

    def test_normalized_to_bbox_normal_batch(self):
        """Normal batch: bbox-sized tensor maps corners and center correctly."""
        keypoints = torch.tensor([
            [[0.0, 0.0]],  # xy for 1 keypoint
            [[1.0, 1.0]],
            [[0.5, 0.5]],
        ])
        bboxes = [
            torch.tensor([0, 0, 100, 200]),   # xyhw
            torch.tensor([20, 30, 100, 200]),
        ]
        for bbox in bboxes:
            kps = normalized_to_bbox(keypoints.clone(), bbox.unsqueeze(0).repeat([3, 1]))
            # (0.0, 0.0) should map to top left corner
            assert kps[0, 0, 0] == bbox[0]
            assert kps[0, 0, 1] == bbox[1]
            # (1.0, 1.0) should map to bottom right corner
            assert kps[1, 0, 0] == bbox[3] + bbox[0]
            assert kps[1, 0, 1] == bbox[2] + bbox[1]
            # (0.5, 0.5) should map to top left corner plus half the new height/width
            assert kps[2, 0, 0] == bbox[3] / 2 + bbox[0]
            assert kps[2, 0, 1] == bbox[2] / 2 + bbox[1]

    def test_normalized_to_bbox_context_batch(self):
        """Context batch: extra bbox edge entries are ignored; middle entries are used."""
        keypoints = torch.tensor([
            [[0.0, 0.0]],
            [[1.0, 1.0]],
            [[0.5, 0.5]],
        ])
        bboxes = [
            torch.tensor([0, 0, 100, 200]),
            torch.tensor([20, 30, 100, 200]),
        ]
        for bbox in bboxes:
            kps = normalized_to_bbox(keypoints.clone(), bbox.unsqueeze(0).repeat([7, 1]))
            # (0.0, 0.0) should map to top left corner
            assert kps[0, 0, 0] == bbox[0]
            assert kps[0, 0, 1] == bbox[1]
            # (1.0, 1.0) should map to bottom right corner
            assert kps[1, 0, 0] == bbox[3] + bbox[0]
            assert kps[1, 0, 1] == bbox[2] + bbox[1]
            # (0.5, 0.5) should map to top left corner plus half the new height/width
            assert kps[2, 0, 0] == bbox[3] / 2 + bbox[0]
            assert kps[2, 0, 1] == bbox[2] / 2 + bbox[1]


class TestConvertBboxCoords:
    """Test the function convert_bbox_coords."""

    def test_convert_bbox_coords_single_view(self, heatmap_data_module):
        """Single-view: cropping the image and adjusting bbox yields identical world coords."""
        # params
        x_crop = 25
        y_crop = 40

        # get training batch
        batch_dict = next(iter(heatmap_data_module.train_dataloader()))
        orig_converted = convert_bbox_coords(batch_dict, batch_dict['keypoints'])
        old_image_dims = [batch_dict['images'].size(-2), batch_dict['images'].size(-1)]
        old_bbox = batch_dict['bbox']
        x_pix = x_crop * old_bbox[:, 3] / old_image_dims[1]
        y_pix = y_crop * old_bbox[:, 2] / old_image_dims[0]

        # create a new batch with smaller & cropped images
        new_dict = batch_dict
        new_dict['images'] = new_dict['images'][:, :, y_crop:-y_crop, x_crop:-x_crop]
        new_dict['bbox'][:, 0] = new_dict['bbox'][:, 0] + x_pix
        new_dict['bbox'][:, 1] = new_dict['bbox'][:, 1] + y_pix
        new_dict['bbox'][:, 2] = new_dict['bbox'][:, 2] - 2 * y_pix
        new_dict['bbox'][:, 3] = new_dict['bbox'][:, 3] - 2 * x_pix
        new_dict['keypoints'][:, 0::2] += x_crop  # keypoints x,y shifted in image
        new_dict['keypoints'][:, 1::2] += y_crop
        new_converted = convert_bbox_coords(new_dict, new_dict['keypoints'])

        # orig and new converted coordinates should be the same
        assert torch.allclose(orig_converted, new_converted, equal_nan=True)

    def test_convert_bbox_coords_multiview(self):
        """Multi-view: each view's keypoints are mapped through their own bbox."""
        batch_dict = {
            'images': torch.tensor(np.random.randn(2, 2, 3, 10, 10)),  # batch, views, RGB, h, w
            'predicted_keypoints': torch.tensor([
                [0.0, 0.0, 0.0, 0.0],  # xy, xy (2 keypoints)
                [10.0, 10.0, 10.0, 10.0],
            ]),
            'bbox': torch.tensor([
                [5.0, 6.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # xyhw x 2
                [0.0, 0.0, 123.0, 124.0, 0.0, 0.0, 3.0, 4.0],
            ]),
            'num_views': torch.tensor([2, 2]),
        }
        converted = convert_bbox_coords(
            batch_dict, batch_dict['predicted_keypoints'],  # type: ignore[arg-type]
        )
        assert converted[0, 0] == batch_dict['bbox'][0, 0]
        assert converted[0, 1] == batch_dict['bbox'][0, 1]
        assert converted[0, 2] == batch_dict['bbox'][0, 4]
        assert converted[0, 3] == batch_dict['bbox'][0, 5]
        assert converted[1, 0] == batch_dict['bbox'][1, 3]
        assert converted[1, 1] == batch_dict['bbox'][1, 2]
        assert converted[1, 2] == batch_dict['bbox'][1, 7]
        assert converted[1, 3] == batch_dict['bbox'][1, 6]

    def test_convert_bbox_coords_multiview_context(self):
        """Multi-view context: edge bbox rows are skipped; middle rows govern conversion."""
        batch_dict = {
            'images': torch.tensor(np.random.randn(2, 2, 3, 10, 10)),  # batch, views, RGB, h, w
            'predicted_keypoints': torch.tensor([
                [0.0, 0.0, 0.0, 0.0],  # xy, xy (2 keypoints)
                [10.0, 10.0, 10.0, 10.0],
            ]),
            'bbox': torch.tensor([
                [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context edge
                [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context edge
                [5.0, 6.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # used
                [0.0, 0.0, 123.0, 124.0, 0.0, 0.0, 3.0, 4.0],         # used
                [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context edge
                [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context edge
            ]),
            'num_views': torch.tensor([2, 2, 2, 2, 2, 2]),
        }
        converted = convert_bbox_coords(
            batch_dict, batch_dict['predicted_keypoints'],  # type: ignore[arg-type]
        )
        assert converted[0, 0] == batch_dict['bbox'][2, 0]
        assert converted[0, 1] == batch_dict['bbox'][2, 1]
        assert converted[0, 2] == batch_dict['bbox'][2, 4]
        assert converted[0, 3] == batch_dict['bbox'][2, 5]
        assert converted[1, 0] == batch_dict['bbox'][3, 3]
        assert converted[1, 1] == batch_dict['bbox'][3, 2]
        assert converted[1, 2] == batch_dict['bbox'][3, 7]
        assert converted[1, 3] == batch_dict['bbox'][3, 6]

    def test_convert_bbox_coords_mismatched_views_raises(self, multiview_heatmap_data_module):
        """Batch elements with different view counts raise ValueError."""
        # get training batch
        batch_dict = next(iter(multiview_heatmap_data_module.train_dataloader()))
        # change number of views for one batch element
        batch_dict['num_views'][0] = 16
        # make sure code complains when batch elements have different numbers of views
        with pytest.raises(ValueError):
            convert_bbox_coords(batch_dict, batch_dict['keypoints'])


class TestConvertOriginalToModelCoords:

    def test_convert_original_to_model_coords_basic(self):
        """Test convert_original_to_model_coords with multiview setup."""

        # Create mock batch_dict
        batch_dict = {
            "num_views": torch.tensor([2, 2]),  # 2 views per batch element
            "images": torch.zeros(2, 2, 3, 256, 256),  # (batch, views, channels, height, width)
            "bbox": torch.tensor([
                # Batch element 0: view 0, view 1
                [0., 0., 100., 200., 50., 25., 100., 200.],
                # Batch element 1: view 0, view 1
                [10., 10., 80., 160., 60., 30., 80., 160.],
            ])
        }

        # Original keypoints: (batch=2, views=2, keypoints=3, xy=2)
        original_keypoints = torch.tensor([
            [  # Batch element 0
                [  # View 0: bbox [0, 0, 100, 200]
                    [0., 0.],  # top-left
                    [200., 100.],  # bottom-right
                    [100., 50.],  # center
                ],
                [  # View 1: bbox [50, 25, 100, 200]
                    [50., 25.],  # top-left
                    [250., 125.],  # bottom-right
                    [150., 75.],  # center
                ]
            ],
            [  # Batch element 1
                [  # View 0: bbox [10, 10, 80, 160]
                    [10., 10.],  # top-left
                    [170., 90.],  # bottom-right
                    [90., 50.],  # center
                ],
                [  # View 1: bbox [60, 30, 80, 160]
                    [60., 30.],  # top-left
                    [220., 110.],  # bottom-right
                    [140., 70.],  # center
                ]
            ]
        ])

        model_keypoints = convert_original_to_model_coords(
            batch_dict, original_keypoints,  # type: ignore[arg-type]
        )

        assert model_keypoints.shape == (2, 2, 3, 2)

        # Top-left corners should be (0, 0)
        assert torch.allclose(model_keypoints[:, :, 0, :], torch.zeros(2, 2, 2), atol=1e-6)

        # Bottom-right corners should be (256, 256)
        assert torch.allclose(model_keypoints[:, :, 1, :], torch.full((2, 2, 2), 256.0), atol=1e-6)

        # Centers should be (128, 128)
        assert torch.allclose(model_keypoints[:, :, 2, :], torch.full((2, 2, 2), 128.0), atol=1e-6)

    def test_convert_original_to_model_coords_different_views(self):
        """Test with different number of views and keypoints."""

        batch_dict = {
            "num_views": torch.tensor([3, 3]),
            "images": torch.zeros(2, 3, 3, 128, 128),  # 128x128 model input
            "bbox": torch.tensor([
                # Batch 0: 3 views with different bboxes
                [0., 0., 50., 100., 25., 25., 50., 100., 50., 50., 50., 100.],
                # Batch 1: 3 views
                [10., 10., 60., 120., 30., 30., 60., 120., 60., 60., 60., 120.],
            ])
        }

        original_keypoints = torch.tensor([
            [  # Batch 0
                [[0., 0.], [100., 50.]],  # View 0: corners of bbox [0,0,50,100]
                [[25., 25.], [125., 75.]],  # View 1: corners of bbox [25,25,50,100]
                [[50., 50.], [150., 100.]],  # View 2: corners of bbox [50,50,50,100]
            ],
            [  # Batch 1
                [[10., 10.], [130., 70.]],  # View 0: corners of bbox [10,10,60,120]
                [[30., 30.], [150., 90.]],  # View 1: corners of bbox [30,30,60,120]
                [[60., 60.], [180., 120.]],  # View 2: corners of bbox [60,60,60,120]
            ]
        ])

        model_keypoints = convert_original_to_model_coords(
            batch_dict, original_keypoints,  # type: ignore[arg-type]
        )

        assert model_keypoints.shape == (2, 3, 2, 2)

        # All top-left corners should map to (0, 0)
        assert torch.allclose(model_keypoints[:, :, 0, :], torch.zeros(2, 3, 2), atol=1e-6)

        # All bottom-right corners should map to (128, 128) since model is 128x128
        assert torch.allclose(model_keypoints[:, :, 1, :], torch.full((2, 3, 2), 128.0), atol=1e-6)


class TestOriginalToModel:

    def test_original_to_model_basic(self):
        """Test original_to_model with basic coordinate transformations."""

        model_width = 256.
        model_height = 256.

        bboxes = [
            torch.tensor([0., 0., 100., 200.]),  # bbox at origin, height=100, width=200
            torch.tensor([50., 25., 100., 200.]),  # bbox offset, same dimensions
        ]

        for bbox in bboxes:

            x, y, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
            keypoints = torch.tensor([
                [[x.item(), y.item()]],           # top-left corner of bbox
                [[x.item() + w.item(), y.item() + h.item()]],  # bottom-right corner of bbox
                [[x.item() + w.item() / 2, y.item() + h.item() / 2]],  # center of bbox
            ])

            kps = original_to_model(
                keypoints.clone(),
                bbox.unsqueeze(0).repeat([3, 1]),
                model_width,
                model_height,
            )

            # Top-left corner of bbox (0,0 in normalized space) should map to (0, 0)
            assert torch.isclose(kps[0, 0, 0], torch.tensor(0.0), atol=1e-6)
            assert torch.isclose(kps[0, 0, 1], torch.tensor(0.0), atol=1e-6)

            # Bottom-right corner should map to (model_width, model_height)
            assert torch.isclose(kps[1, 0, 0], torch.tensor(model_width), atol=1e-6)
            assert torch.isclose(kps[1, 0, 1], torch.tensor(model_height), atol=1e-6)

            # Center should map to (model_width/2, model_height/2)
            assert torch.isclose(kps[2, 0, 0], torch.tensor(model_width / 2), atol=1e-6)
            assert torch.isclose(kps[2, 0, 1], torch.tensor(model_height / 2), atol=1e-6)

    def test_original_to_model_context_batch(self):
        """Test original_to_model with context batch (extra bbox entries for edges)."""

        model_width = 256.
        model_height = 256.

        bboxes = [
            torch.tensor([0., 0., 100., 200.]),
            torch.tensor([50., 25., 100., 200.]),
        ]

        for bbox in bboxes:

            x, y, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
            keypoints = torch.tensor([
                [[x.item(), y.item()]],
                [[x.item() + w.item(), y.item() + h.item()]],
                [[x.item() + w.item() / 2, y.item() + h.item() / 2]],
            ])

            # 7-entry bbox tensor for context batch
            bbox_context = bbox.unsqueeze(0).repeat([7, 1])

            kps = original_to_model(
                keypoints.clone(),
                bbox_context,
                model_width,
                model_height,
            )

            assert torch.isclose(kps[0, 0, 0], torch.tensor(0.0), atol=1e-6)
            assert torch.isclose(kps[0, 0, 1], torch.tensor(0.0), atol=1e-6)

            assert torch.isclose(kps[1, 0, 0], torch.tensor(model_width), atol=1e-6)
            assert torch.isclose(kps[1, 0, 1], torch.tensor(model_height), atol=1e-6)

            assert torch.isclose(kps[2, 0, 0], torch.tensor(model_width / 2), atol=1e-6)
            assert torch.isclose(kps[2, 0, 1], torch.tensor(model_height / 2), atol=1e-6)

    def test_original_to_model_different_dimensions(self):
        """Test with non-square model dimensions."""

        keypoints = torch.tensor([
            [[50., 25.]],  # top-left of bbox
            [[150., 75.]],  # bottom-right of bbox
            [[100., 50.]],  # center of bbox
        ])

        bbox = torch.tensor([50., 25., 50., 100.])  # x=50, y=25, h=50, w=100
        model_width = 128.
        model_height = 64.

        kps = original_to_model(
            keypoints,
            bbox.unsqueeze(0).repeat([3, 1]),
            model_width,
            model_height,
        )

        # Top-left: (50-50)/100 * 128 = 0, (25-25)/50 * 64 = 0
        assert torch.isclose(kps[0, 0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(kps[0, 0, 1], torch.tensor(0.0), atol=1e-6)

        # Bottom-right: (150-50)/100 * 128 = 128, (75-25)/50 * 64 = 64
        assert torch.isclose(kps[1, 0, 0], torch.tensor(128.0), atol=1e-6)
        assert torch.isclose(kps[1, 0, 1], torch.tensor(64.0), atol=1e-6)

        # Center: (100-50)/100 * 128 = 64, (50-25)/50 * 64 = 32
        assert torch.isclose(kps[2, 0, 0], torch.tensor(64.0), atol=1e-6)
        assert torch.isclose(kps[2, 0, 1], torch.tensor(32.0), atol=1e-6)
