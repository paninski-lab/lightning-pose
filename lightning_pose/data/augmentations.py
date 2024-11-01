"""Functions to build augmentation pipeline."""

import imgaug.augmenters as iaa
from omegaconf import DictConfig
from typeguard import typechecked

# to ignore imports for sphix-autoapidoc
__all__ = [
    "imgaug_transform",
]


@typechecked
def imgaug_transform(cfg: DictConfig) -> iaa.Sequential:
    """Create simple data transform pipeline that augments images.

    Args:
        cfg: standard config file that carries around dataset info; relevant is the parameter
            "cfg.training.imgaug" which can take on the following values:
            - default: resizing only
            - dlc: imgaug pipeline implemented in DLC 2.0 package
            - dlc-top-down: `dlc` pipeline plus random flipping along both horizontal and vertical
              axes

    Returns:
        imgaug pipeline

    """

    kind = cfg.training.get("imgaug", "default")

    data_transform = []

    if kind == "default":

        # resizing happens below
        print("using default image augmentation pipeline (resizing only)")

    elif kind == "dlc" or kind == "dlc-top-down":

        print(f"using {kind} image augmentation pipeline")

        # flip around horizontal/vertical axes first
        if kind == "dlc-top-down":
            # vertical axis
            data_transform.append(iaa.Fliplr(p=0.5))
            # horizontal axis
            data_transform.append(iaa.Flipud(p=0.5))

        apply_prob_0 = 0.5

        # rotate
        rotation = 25  # rotation uniformly sampled from (-rotation, +rotation)
        data_transform.append(iaa.Sometimes(
            0.4,
            iaa.Affine(rotate=(-rotation, rotation))
        ))
        # motion blur
        k = 5  # kernel size of blur
        angle = 90  # blur direction uniformly sampled from (-angle, +angle)
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.MotionBlur(k=k, angle=(-angle, angle))
        ))
        # coarse dropout
        prct = 0.02  # drop `prct` of all pixels by converting them to black pixels
        size_prct = 0.3  # drop pix on a low-res version of img that's `size_prct` of og
        per_channel = 0.5  # per_channel transformations on `per_channel` frac of images
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.CoarseDropout(p=prct, size_percent=size_prct, per_channel=per_channel)
        ))
        # coarse salt and pepper
        # bright reflections can often confuse the model into thinking they are paws
        # (which can also just be bright blobs) - so include some additional transforms that put
        # bright blobs (and dark blobs) into the image
        prct = 0.01  # probability of changing a pixel to salt/pepper noise
        size_prct = (0.05, 0.1)  # drop pix on a low-res version of img that's `size_prct` of og
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.CoarseSalt(p=prct, size_percent=size_prct),  # bigger chunks than coarse dropout
        ))
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.CoarsePepper(p=prct, size_percent=size_prct),
        ))
        # elastic transform
        alpha = (0, 10)  # controls strength of displacement
        sigma = 5  # cotnrols smoothness of displacement
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
        ))
        # hist eq
        data_transform.append(iaa.Sometimes(
            0.1,
            iaa.AllChannelsHistogramEqualization()
        ))
        # clahe (contrast limited adaptive histogram equalization) -
        # hist eq over image patches
        data_transform.append(iaa.Sometimes(
            0.1,
            iaa.AllChannelsCLAHE()
        ))
        # emboss
        alpha = (0, 0.5)  # overlay embossed image on original with alpha in this range
        strength = (0.5, 1.5)  # strength of embossing lies in this range
        data_transform.append(iaa.Sometimes(
            0.1,
            iaa.Emboss(alpha=alpha, strength=strength)
        ))
        # crop
        crop_by = 0.15  # number of pix to crop on each side of img given as a fraction
        data_transform.append(iaa.Sometimes(
            0.4,
            iaa.CropAndPad(percent=(-crop_by, crop_by), keep_size=False)
        ))

    else:
        raise NotImplementedError("must choose imgaug kind from 'default', 'dlc', 'dlc-top-down'")

    # do not resize when using dynamic crop pipeline
    if not cfg.data.get('dynamic_crop', False):
        data_transform.append(
            iaa.Resize({
                "height": cfg.data.image_resize_dims.height,
                "width": cfg.data.image_resize_dims.width}
            ))

    return iaa.Sequential(data_transform)
