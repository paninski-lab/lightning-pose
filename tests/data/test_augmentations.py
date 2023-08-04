"""Test imgaug pipeline functionality."""

import copy
import os

import numpy as np
import pytest
from PIL import Image

from lightning_pose.data.augmentations import imgaug_transform


def test_imgaug_transform_default(cfg, base_dataset):

    cfg_tmp = copy.deepcopy(cfg)

    idx = 0
    img_name = base_dataset.image_names[idx]
    keypoints_on_image = base_dataset.keypoints[idx]
    file_name = os.path.join(base_dataset.root_directory, img_name)
    image = Image.open(file_name).convert("RGB")

    # default pipeline: resize only
    cfg_tmp.training.imgaug = "default"
    pipe = imgaug_transform(cfg_tmp)
    im_0, kps_0 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints_on_image, axis=0),
    )
    im_0 = im_0[0]
    kps_0 = kps_0[0].reshape(-1)
    assert im_0.shape[0] == cfg.data.image_resize_dims.height
    assert im_0.shape[1] == cfg.data.image_resize_dims.width

    # default pipeline: should be repeatable
    im_1, kps_1 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints_on_image, axis=0),
    )
    im_1 = im_1[0]
    kps_1 = kps_1[0].reshape(-1)
    assert np.allclose(im_0, im_1)
    assert np.allclose(kps_0, kps_1, equal_nan=True)

    # invalid pipeline: ensure error is raised
    cfg_tmp.training.imgaug = "null"
    with pytest.raises(NotImplementedError):
        imgaug_transform(cfg_tmp)


def test_imgaug_transform_dlc(cfg, base_dataset):

    cfg_tmp = copy.deepcopy(cfg)

    idx = 0
    img_name = base_dataset.image_names[idx]
    keypoints_on_image = base_dataset.keypoints[idx]
    file_name = os.path.join(base_dataset.root_directory, img_name)
    image = Image.open(file_name).convert("RGB")

    # dlc pipeline: should not contain flips
    cfg_tmp.training.imgaug = "dlc"
    pipe = imgaug_transform(cfg_tmp)
    assert pipe.__str__().find("Fliplr") == -1
    assert pipe.__str__().find("Flipud") == -1

    # dlc pipeline: should resize
    im_0, kps_0 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints_on_image, axis=0),
    )
    im_0 = im_0[0]
    assert im_0.shape[0] == cfg.data.image_resize_dims.height
    assert im_0.shape[1] == cfg.data.image_resize_dims.width

    # dlc pipeline: should not be repeatable
    im_1, kps_1 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints_on_image, axis=0),
    )
    im_1 = im_1[0]
    assert not np.allclose(im_0, im_1)


def test_imgaug_transform_dlc_top_down(cfg, base_dataset):

    cfg_tmp = copy.deepcopy(cfg)

    idx = 0
    img_name = base_dataset.image_names[idx]
    keypoints_on_image = base_dataset.keypoints[idx]
    file_name = os.path.join(base_dataset.root_directory, img_name)
    image = Image.open(file_name).convert("RGB")

    # dlc-top-down pipeline: should contain flips
    cfg_tmp.training.imgaug = "dlc-top-down"
    pipe = imgaug_transform(cfg_tmp)
    assert pipe.__str__().find("Fliplr") != -1
    assert pipe.__str__().find("Flipud") != -1

    # dlc-top-down pipeline: should resize
    im_0, kps_0 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints_on_image, axis=0),
    )
    im_0 = im_0[0]
    assert im_0.shape[0] == cfg.data.image_resize_dims.height
    assert im_0.shape[1] == cfg.data.image_resize_dims.width

    # dlc-top-down pipeline: should not be repeatable
    im_1, kps_1 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints_on_image, axis=0),
    )
    im_1 = im_1[0]
    assert not np.allclose(im_0, im_1)
