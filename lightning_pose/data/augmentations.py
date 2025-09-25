"""Functions to build augmentation pipeline."""

from typing import Any

import imgaug.augmenters as iaa
from omegaconf import DictConfig, ListConfig
from typeguard import typechecked

# to ignore imports for sphix-autoapidoc
__all__ = [
    "imgaug_transform",
]


@typechecked
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
        transform = getattr(iaa, transform_str)
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


def expand_imgaug_str_to_dict(params: str) -> dict[str, Any]:

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
