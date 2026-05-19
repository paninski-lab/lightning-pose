"""Test DataExtractor functionality."""

from lightning_pose.data.extractor import DataExtractor


class TestDataExtractor:
    """Test the DataExtractor class."""

    def test_data_extractor_single_view(self, base_data_module_combined):
        num_frames = (
            len(base_data_module_combined.dataset)
            * base_data_module_combined.train_probability
        )
        keypoint_tensor, _ = DataExtractor(
            data_module=base_data_module_combined, cond='train'
        )()
        assert keypoint_tensor.shape == (num_frames, 34)

        keypoint_tensor, images_tensor = DataExtractor(
            data_module=base_data_module_combined, cond='train', extract_images=True
        )()
        assert images_tensor is not None
        assert images_tensor.shape == (num_frames, 3, 128, 128)

    def test_data_extractor_multiview(self, multiview_heatmap_data_module_combined):
        num_frames = (
            len(multiview_heatmap_data_module_combined.dataset)
            * multiview_heatmap_data_module_combined.train_probability
        )
        keypoint_tensor, _ = DataExtractor(
            data_module=multiview_heatmap_data_module_combined, cond='train'
        )()
        assert keypoint_tensor.shape == (num_frames, 28)
