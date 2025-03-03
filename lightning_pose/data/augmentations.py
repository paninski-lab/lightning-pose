"""Functions to build augmentation pipeline."""

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
            e.g. "Affine", "Fliplr", etc. The value must be a dictionary with two keys: "p" is the
            probability of applying the transform (using imgaug.augmenters.Sometimes) and
            "kwargs" is a dictionary of keyword args sent to that specific augmentation.

    Examples:

    1. Create a pipeline with
        - Affine transformation applied 50% of the time with rotation uniformly sampled from
          (-25, 25) degrees
        - MotionBlur transformation that is applied 25% of the time with a kernel size of 5 pixels
          and blur direction uniformly sampled from (-90, 90) degrees

        >>> params_dict{
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

    Returns:
        imgaug pipeline

    """

    data_transform = []

    for transform_str, args in params_dict.items():

        transform = getattr(iaa, transform_str)
        apply_prob = args.get("p", 0.5)
        transform_args = args.get("args", ())
        transform_kwargs = args.get("kwargs", {})

        # make sure any lists are converted to tuples; DictConfig cannot load tuples from yaml
        # files, but no iaa args are lists
        for kw, arg in transform_kwargs.items():
            if isinstance(arg, list) or isinstance(arg, ListConfig):
                transform_kwargs[kw] = tuple(arg)

        # add transform to pipeline
        if apply_prob == 0.0:
            pass
        elif apply_prob < 1.0:
            data_transform.append(iaa.Sometimes(
                apply_prob,
                transform(*transform_args, **transform_kwargs),
            ))
        else:
            data_transform.append(transform(*transform_args, **transform_kwargs))

    return iaa.Sequential(data_transform)
