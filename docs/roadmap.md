# Lightning Pose development roadmap

## General enhancements
- [x] multi-GPU training for supervised models ([#206](https://github.com/paninski-lab/lightning-pose/pull/206))
- [x] multi-GPU training for unsupervised models ([#207](https://github.com/paninski-lab/lightning-pose/pull/207))
- [ ] introduce jaxtyping (see [here](https://github.com/google/jaxtyping/issues/70))

## Losses and backbones
- [x] incorporate transformer backbones ([#84](https://github.com/danbider/lightning-pose/pull/84), [#106](https://github.com/danbider/lightning-pose/pull/106))
- [ ] compute non-temporal unsupervised losses on labeled data

## Multi-view support for non-mirrored setups
- [x] implement supervised datasets/dataloaders that work with multiple views ([#115](https://github.com/danbider/lightning-pose/pull/115))
- [x] context frames for multi-view ([#126](https://github.com/danbider/lightning-pose/pull/126))
- [x] unsupervised losses for multi-view ([#187](https://github.com/danbider/lightning-pose/pull/187))

## Single-view dynamic crop (small animals in large frames)
- [ ] implement dynamic cropping pipeline with detector model and pose estimator
- [ ] context frames for dynamic crop
- [ ] unsupervised losses for dynamic crop

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
