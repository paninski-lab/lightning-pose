"""Test imgaug pipeline functionality."""

import os

import numpy as np
import pytest
from PIL import Image

from lightning_pose.data.augmentations import imgaug_transform


class TestImgaugTransform:

    @pytest.fixture
    def image(self, base_dataset):
        idx = 0
        file_name = os.path.join(base_dataset.root_directory, base_dataset.image_names[idx])
        return Image.open(file_name).convert("RGB")

    @pytest.fixture
    def keypoints(self, base_dataset):
        idx = 0
        return base_dataset.keypoints[idx]

    def test_empyt_params_dict(self, image, keypoints):
        """Pipeline should not do anything if params_dict is empty"""
        params_dict = {}
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        kps_0 = kps_0[0]
        assert np.allclose(image, im_0)
        assert np.allclose(keypoints, kps_0, equal_nan=True)

    def test_zero_probabilities(self, image, keypoints):
        """Pipeline should not do anything if augmentation probabilities are all zero"""
        params_dict = {
            "ShearX": {"p": 0.0, "kwargs": {"shear": (-30, 30)}},
            "Jigsaw": {"p": 0.0, "kwargs": {"nb_rows": (3, 10), "nb_cols": (5, 8)}},
            "MultiplyAndAddToBrightness": {"p": 0., "kwargs": {"mul": (0.5, 1.5), "add": (-5, 5)}},
        }
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        kps_0 = kps_0[0]
        assert np.allclose(image, im_0)
        assert np.allclose(keypoints, kps_0, equal_nan=True)

    def test_resize_shape(self, image, keypoints):
        """Resize should properly resize images"""
        params_dict = {
            "Resize": {"p": 1.0, "args": ({"height": 256, "width": 256},), "kwargs": {}}
        }
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        assert im_0.shape[0] == params_dict["Resize"]["args"][0]["height"]
        assert im_0.shape[1] == params_dict["Resize"]["args"][0]["width"]

    def test_resize_repeatable(self, image, keypoints):
        """Resize should be repeatable"""
        params_dict = {
            "Resize": {"p": 1.0, "args": ({"height": 256, "width": 256},), "kwargs": {}}
        }
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]

        im_1, kps_1 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_1 = im_1[0]
        kps_1 = kps_1[0]
        assert np.allclose(im_0, im_1)
        assert np.allclose(kps_0, kps_1, equal_nan=True)

    def test_fliplr(self, image, keypoints):
        """Fliplr should flip both image and keypoints"""
        params_dict = {"Fliplr": {"p": 1.0, "kwargs": {"p": 1.0}}}
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        im_0 = im_0[:, ::-1, ...]  # lr flip
        kps_0 = kps_0[0]
        kps_0[:, 0] = image.size[0] - kps_0[:, 0]  # lr flip; PIL.Image.size is (width, height)
        assert np.allclose(im_0, image)
        assert np.allclose(kps_0, keypoints, equal_nan=True)

    def test_flipud(self, image, keypoints):
        """Flipud should flip both image and keypoints"""
        params_dict = {"Flipud": {"p": 1.0, "kwargs": {"p": 1.0}}}
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        im_0 = im_0[::-1, :, ...]  # ud flip
        kps_0 = kps_0[0]
        kps_0[:, 1] = image.size[1] - kps_0[:, 1]  # ud flip; PIL.Image.size is (width, height)
        assert np.allclose(im_0, image)
        assert np.allclose(kps_0, keypoints, equal_nan=True)

    def test_motionblur(self, image, keypoints):
        """MotionBlur should alter image but not keypoints"""
        params_dict = {"MotionBlur": {"p": 1.0, "kwargs": {"k": 5, "angle": (-90, 90)}}}
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        kps_0 = kps_0[0]
        assert not np.allclose(im_0, image)  # image changed
        assert np.allclose(kps_0, keypoints, equal_nan=True)  # keypoints do not

    def test_coarsesalt(self, image, keypoints):
        """CoarseSalt should alter image but not keypoints"""
        params_dict = {"CoarseSalt": {"p": 1.0, "kwargs": {"p": 0.1, "size_percent": (0.05, 1.0)}}}
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        kps_0 = kps_0[0]
        assert not np.allclose(im_0, image)  # image changed
        assert np.allclose(kps_0, keypoints, equal_nan=True)  # keypoints do not

    def test_affine(self, image, keypoints):
        """Affine should alter image and keypoints"""
        params_dict = {"Affine": {"p": 1.0, "kwargs": {"rotate": (-90, 90)}}}
        pipe = imgaug_transform(params_dict)
        im_0, kps_0 = pipe(
            images=np.expand_dims(image, axis=0),
            keypoints=np.expand_dims(keypoints, axis=0),
        )
        im_0 = im_0[0]
        kps_0 = kps_0[0]
        assert not np.allclose(im_0, image)  # image changed
        assert not np.allclose(kps_0, keypoints, equal_nan=True)  # keypoints changed

    def test_rot90(self, image, keypoints):
        """Test different option configurations for rot90"""
        params_dict = {"Rot90": {"p": 1.0, "kwargs": {"k": [[0, 1]]}}}
        pipe = imgaug_transform(params_dict)
        assert pipe.__str__().find("Choice(a=[0, 1]") > -1

        params_dict = {"Rot90": {"p": 1.0, "kwargs": {"k": [[0, 1, 4]]}}}
        pipe = imgaug_transform(params_dict)
        assert pipe.__str__().find("Choice(a=[0, 1, 4]") > -1

        params_dict = {"Rot90": {"p": 1.0, "kwargs": {"k": [0, 1, 4]}}}
        pipe = imgaug_transform(params_dict)
        assert pipe.__str__().find("Choice(a=[0, 1, 4]") > -1

        params_dict = {"Rot90": {"p": 1.0, "kwargs": {"k": [0, 3]}}}
        pipe = imgaug_transform(params_dict)
        assert pipe.__str__().find(
            "param=DiscreteUniform(Deterministic(int 0), Deterministic(int 3))"
        ) > -1
