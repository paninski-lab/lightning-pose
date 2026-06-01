# Lightning Pose development roadmap

## General enhancements
- [ ] support for training/inference with mixed precision
- [ ] introduce `visibility` column in labeled data to match COCO format; treat occluded and unlabeled points differently ([Issue](https://github.com/paninski-lab/lightning-pose/issues/263))
- [ ] read COCO JSON label files (need to address `visibility` handling first)
- [x] introduce jaxtyping ([#407](https://github.com/paninski-lab/lightning-pose/pull/407))
- [x] multi-GPU training for unsupervised models ([#207](https://github.com/paninski-lab/lightning-pose/pull/207))
- [x] multi-GPU training for supervised models ([#206](https://github.com/paninski-lab/lightning-pose/pull/206))

## Video reading enhancements
- [ ] look into using [pynvvideocodec](https://developer.nvidia.com/pynvvideocodec) for accelerated inference and Windows support
- [ ] add OpenCV video reader option for inference for native Windows compatability

## Losses and backbones
- [ ] compute non-temporal unsupervised losses on labeled data
- [x] incorporate transformer backbones ([#84](https://github.com/paninski-lab/lightning-pose/pull/84), [#106](https://github.com/paninski-lab/lightning-pose/pull/106))

## Multi-view support for non-mirrored setups
- [x] unsupervised losses for multi-view ([#187](https://github.com/paninski-lab/lightning-pose/pull/187))
- [x] context frames for multi-view ([#126](https://github.com/paninski-lab/lightning-pose/pull/126))
- [x] implement supervised datasets/dataloaders that work with multiple views ([#115](https://github.com/paninski-lab/lightning-pose/pull/115))

## Single-view dynamic crop (small animals in large frames)
- [ ] split CLI `crop` command into `create_bbox` and `smooth_bbox` commands for modularity ([Issue](https://github.com/paninski-lab/lightning-pose/issues/415))
- [x] context frames for dynamic crop ([#250](https://github.com/paninski-lab/lightning-pose/pull/250))
- [x] unsupervised losses for dynamic crop ([#250](https://github.com/paninski-lab/lightning-pose/pull/250))
- [x] implement dynamic cropping pipeline with detector model and pose estimator ([#250](https://github.com/paninski-lab/lightning-pose/pull/250))

## Multi-view dynamic crop
- [ ] perform view-specific dynamic cropping, re-assemble views after pose estimation stage
- [ ] context frames for multi-view dynamic crop
- [ ] unsupervised losses for multi-view dynamic crop

## Multi-animal
- [ ] single-view, supervised
- [ ] single-view, context
- [ ] single-view, unsupervised losses
- [ ] multi-view, supervised
- [ ] multi-view, context
- [ ] multi-view, unsupervised losses
