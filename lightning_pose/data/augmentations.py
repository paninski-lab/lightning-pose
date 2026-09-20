"""Functions to build augmentation pipeline."""

from typing import Any

import imgaug.augmenters as iaa
from omegaconf import DictConfig, ListConfig

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def imgaug_transform(params_dict: dict | DictConfig) -> iaa.Sequential:
    """Create simple and flexible data transform pipeline that augments images and keypoints.

    Args:
        params_dict: each key must be the name of a transform importable from imgaug.augmenters,
            e.g. "Affine", "Fliplr", etc. The value must be a dict with several optional keys:
            - "p" (float): probability of applying transform (using imgaug.augmenters.Sometimes)
            - "args" (list): arguments for transform
            - "kwargs" (dict): keyword args for the transformation

    Examples:

        Create a pipeline with
        - Affine transformation applied 50% of the time with rotation uniformly sampled from
          (-25, 25) degrees
        - MotionBlur transformation that is applied 25% of the time with a kernel size of 5 pixels
          and blur direction uniformly sampled from (-90, 90) degrees

        >>> params_dict = {
        >>>    'Affine': {'p': 0.5, 'kwargs': {'rotate': (-25, 25)}},
        >>>    'MotionBlur': {'p': 0.25, 'kwargs': {'k': 5, 'angle': (-90, 90)}},
        >>> }

        In a config file, this will look like:
        >>> training:
        >>>   imgaug:
        >>>     Affine:
        >>>       p: 0.5
        >>>       kwargs:
        >>>         rotate: [-10, 10]
        >>>     MotionBlur:
        >>>       p: 0.25
        >>>       kwargs:
        >>>         k: 5
        >>>         angle: [-90, 90]

        Create a pipeline with
        - Rot90 transformation applied 100% of the time with rotations of 0, 90, 180, 270 degrees.

        >>> params_dict = {
        >>>     'Rot90': {'p': 1.0, 'kwargs': {'k': [[0, 1, 2, 3]]}},  # note required nested list
        >>> }

        In a config file, this will look like:
        >>> training:
        >>>   imgaug:
        >>>     Rot90:
        >>>       p: 1.0
        >>>       kwargs:
        >>>         k: [0, 1, 2, 3]

        NOTE: if you pass a list of exactly 2 values to Rot90 it will be parsed as a tuple and all
        (discrete) rotations between the two values will be sampled uniformly.
        For example, `k: [0, 2]` is equivalent to `k: [0, 1, 2]`.
        If you need to _only_ sample two non-contiguous integers please raise an issue.

    Returns:
        imgaug pipeline

    """

    data_transform = []

    for transform_str, args in params_dict.items():
        if str(transform_str) == 'Downscale':
            transform = downscale
        else:
            transform = getattr(iaa, str(transform_str))
        apply_prob = args.get("p", 0.5)
        transform_args = args.get("args", ())
        transform_kwargs = args.get("kwargs", {})

        # DictConfig cannot load tuples from yaml files
        # make sure any lists are converted to tuples
        # unless the list contains a single item, then pass through the item (hack for Rot90)
        for kw, arg in transform_kwargs.items():
            if isinstance(arg, list) or isinstance(arg, ListConfig):
                if len(arg) == 1:
                    transform_kwargs[kw] = arg[0]
                elif len(arg) == 2:
                    transform_kwargs[kw] = tuple(arg)
                else:
                    transform_kwargs[kw] = arg

        # add transform to pipeline
        if apply_prob == 0.0:
            pass
        elif apply_prob < 1.0:
            data_transform.append(
                iaa.Sometimes(
                    apply_prob,
                    transform(*transform_args, **transform_kwargs),
                )
            )
        else:
            data_transform.append(transform(*transform_args, **transform_kwargs))

    return iaa.Sequential(data_transform)


def downscale(
    scale: tuple[float, float] = (0.3, 1.0),
    interpolation: str = 'area',
) -> iaa.Lambda:
    """Resolution augmentation: shrink each image by a random factor and stretch it back.

    Each image is resized by a factor sampled uniformly from ``scale`` and then resized back to
    its original height and width, so the geometry (and therefore every keypoint and heatmap) is
    unchanged while fine detail is lost, as when a small source frame is enlarged by a zoom-in
    crop. Motivation: in a multi-dataset corpus the same body part is supervised at very
    different native resolutions; without this, "sharp and large" and "blurred and large" are
    distinguishable appearances and a channel can bind to one of them (the facemap pupil
    problem, 2026-09-20). Not an imgaug built-in, hence built from ``iaa.Lambda``; the config
    key is ``Downscale`` (``p``, ``kwargs: {scale: [lo, hi]}``), handled by
    :func:`imgaug_transform`.

    Args:
        scale: (lower, upper) bounds of the shrink factor, sampled per image; 1.0 = unchanged.
        interpolation: OpenCV interpolation used for the shrink ('area' or 'linear'); the
            stretch back always uses linear interpolation.

    Returns:
        an imgaug augmenter that only touches images.

    """
    import cv2
    import numpy as np

    lo, hi = float(scale[0]), float(scale[1])
    if not 0.0 < lo <= hi <= 1.0:
        raise ValueError(f'Downscale scale must satisfy 0 < lower <= upper <= 1, got {scale}')
    shrink_interp = cv2.INTER_AREA if interpolation == 'area' else cv2.INTER_LINEAR

    def func_images(images, random_state, parents, hooks):
        out = []
        for image in images:
            s = float(random_state.uniform(lo, hi))
            h, w = image.shape[:2]
            hs, ws = max(int(round(h * s)), 2), max(int(round(w * s)), 2)
            if hs >= h and ws >= w:
                out.append(image)
                continue
            small = cv2.resize(image, (ws, hs), interpolation=shrink_interp)
            back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            if back.ndim == 2 and image.ndim == 3:
                back = back[..., None]
            out.append(np.ascontiguousarray(back, dtype=image.dtype))
        return out

    # keypoints, heatmaps, bounding boxes are geometrically unchanged: leave them untouched
    return iaa.Lambda(func_images=func_images, name='Downscale')


def expand_imgaug_str_to_dict(params: str) -> dict[str, Any]:
    """Expand a shorthand augmentation string to a full parameter dictionary.

    Args:
        params: augmentation preset string. One of ``"default"``, ``"none"``, ``"dlc"``,
            ``"dlc-lr"``, ``"dlc-top-down"``, or ``"dlc-mv"``.

    Returns:
        Dictionary mapping augmentation transform names to their parameter dicts, suitable
        for passing to :func:`imgaug_transform`.

    Raises:
        NotImplementedError: if ``params`` is not one of the allowed preset strings.
    """

    _allowed_imgaug_strs = [
        "default",
        "none",
        "dlc",
        "dlc-lr",
        "dlc-top-down",
        "dlc-mv",
    ]

    params_dict = {}
    if params in ["default", "none"]:
        pass  # no augmentations
    elif params in ["dlc", "dlc-lr", "dlc-top-down", "dlc-mv"]:

        # rotate 0 or 180 degrees
        if params in ["dlc-lr"]:
            params_dict["Rot90"] = {"p": 1.0, "kwargs": {"k": [[0, 2]]}}

        # rotate 0, 90, 180, or 270 degrees
        if params in ["dlc-top-down"]:
            params_dict["Rot90"] = {"p": 1.0, "kwargs": {"k": [[0, 1, 2, 3]]}}

        # rotate
        if not params.endswith("mv"):
            rotation = 25  # rotation uniformly sampled from (-rotation, +rotation)
            params_dict["Affine"] = {"p": 0.4, "kwargs": {"rotate": (-rotation, rotation)}}

        # motion blur
        k = 5  # kernel size of blur
        angle = 90  # blur direction uniformly sampled from (-angle, +angle)
        params_dict["MotionBlur"] = {
            "p": 0.5,
            "kwargs": {"k": k, "angle": (-angle, angle)},
        }

        # coarse dropout
        prct = 0.02  # drop `prct` of all pixels by converting them to black pixels
        size_prct = 0.3  # drop pix on a low-res version of img that's `size_prct` of og
        per_channel = 0.5  # per_channel transformations on `per_channel` frac of images
        params_dict["CoarseDropout"] = {
            "p": 0.5,
            "kwargs": {
                "p": prct,
                "size_percent": size_prct,
                "per_channel": per_channel,
            },
        }

        # coarse salt and pepper
        # bright reflections can often confuse the model into thinking they are paws
        # (which can also just be bright blobs) - so include some additional transforms that
        # put bright blobs (and dark blobs) into the image
        # bigger chunks than coarse dropout
        prct = 0.01  # probability of changing a pixel to salt/pepper noise
        size_prct = (
            0.05,
            0.1,
        )  # drop pix on low-res version of img that's `size_prct` of og
        params_dict["CoarseSalt"] = {
            "p": 0.5,
            "kwargs": {"p": prct, "size_percent": size_prct},
        }
        params_dict["CoarsePepper"] = {
            "p": 0.5,
            "kwargs": {"p": prct, "size_percent": size_prct},
        }

        # elastic transform
        if not params.endswith("mv"):
            alpha = (0, 10)  # controls strength of displacement
            sigma = 5  # cotnrols smoothness of displacement
            params_dict["ElasticTransformation"] = {
                "p": 0.5,
                "kwargs": {"alpha": alpha, "sigma": sigma},
            }

        # hist eq
        params_dict["AllChannelsHistogramEqualization"] = {"p": 0.1, "kwargs": {}}

        # clahe (contrast limited adaptive histogram equalization) -
        # hist eq over image patches
        params_dict["AllChannelsCLAHE"] = {"p": 0.1, "kwargs": {}}

        # emboss
        alpha = (0, 0.5)  # overlay embossed image on original with alpha in this range
        strength = (0.5, 1.5)  # strength of embossing lies in this range
        params_dict["Emboss"] = {
            "p": 0.1,
            "kwargs": {"alpha": alpha, "strength": strength},
        }

        # crop
        if not params.endswith("mv"):
            crop_by = 0.15  # number of pix to crop on each side of img given as a fraction
            params_dict["CropAndPad"] = {
                "p": 0.4,
                "kwargs": {"percent": (-crop_by, crop_by), "keep_size": False},
            }
    else:
        raise NotImplementedError(
            f"cfg.training.imgaug string {params} must be in {_allowed_imgaug_strs}"
        )

    return params_dict
