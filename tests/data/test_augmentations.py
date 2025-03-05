"""Test imgaug pipeline functionality."""

import copy
import os

import numpy as np
import pytest
from PIL import Image

from lightning_pose.data.augmentations import imgaug_transform


def test_imgaug_transform(base_dataset):

    idx = 0
    img_name = base_dataset.image_names[idx]
    keypoints = base_dataset.keypoints[idx]
    file_name = os.path.join(base_dataset.root_directory, img_name)
    image = Image.open(file_name).convert("RGB")

    # play with several easy-to-verify transforms

    # ------------
    # NULL
    # ------------
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

    # pipeline should not do anything if augmentation probabilities are all zero
    params_dict = {
        "ShearX": {"p": 0.0, "kwargs": {"shear": (-30, 30)}},
        "Jigsaw": {"p": 0.0, "kwargs": {"nb_rows": (3, 10), "nb_cols": (5, 8)}},
        "MultiplyAndAddToBrightness": {"p": 0.0, "kwargs": {"mul": (0.5, 1.5), "add": (-5, 5)}},
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

    # ------------
    # Resize
    # ------------
    params_dict = {"Resize": {"p": 1.0, "args": ({"height": 256, "width": 256},), "kwargs": {}}}
    pipe = imgaug_transform(params_dict)
    im_0, kps_0 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints, axis=0),
    )
    im_0 = im_0[0]
    assert im_0.shape[0] == params_dict["Resize"]["args"][0]["height"]
    assert im_0.shape[1] == params_dict["Resize"]["args"][0]["width"]

    # resize should be repeatable
    im_1, kps_1 = pipe(
        images=np.expand_dims(image, axis=0),
        keypoints=np.expand_dims(keypoints, axis=0),
    )
    im_1 = im_1[0]
    kps_1 = kps_1[0]
    assert np.allclose(im_0, im_1)
    assert np.allclose(kps_0, kps_1, equal_nan=True)

    # ------------
    # Fliplr
    # ------------
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

    # ------------
    # Flipud
    # ------------
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

    # ------------
    # misc
    # ------------
    # make sure various augmentations are not repeatable
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
