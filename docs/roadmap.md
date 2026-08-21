# Lightning Pose development roadmap

## General enhancements
- [x] introduce jaxtyping ([#407](https://github.com/paninski-lab/lightning-pose/pull/407))
- [x] multi-GPU training for unsupervised models ([#207](https://github.com/paninski-lab/lightning-pose/pull/207))
- [x] multi-GPU training for supervised models ([#206](https://github.com/paninski-lab/lightning-pose/pull/206))
- [x] introduce `visibility` column in labeled data to match COCO format; treat occluded and unlabeled points differently ([#440](https://github.com/paninski-lab/lightning-pose/pull/440))
- [ ] read COCO-style JSON label files
- [ ] export predictions in COCO-style JSON format
- [ ] support training across multiple datasets; see [#284](https://github.com/paninski-lab/lightning-pose/pull/284)
- [ ] training/inference support on macOS (no semi-supervised models that require DALI)
- [ ] training/inference support on Windows (no semi-supervised models that require DALI)

## Inference enhancements
- [x] inference with mixed precision ([#458](https://github.com/paninski-lab/lightning-pose/pull/458))
- [x] `torch.compile()` to improve inference speed ([#469](https://github.com/paninski-lab/lightning-pose/pull/469))
- [x] [ONNX runtime](https://onnxruntime.ai/) to improve inference speed ([#471](https://github.com/paninski-lab/lightning-pose/pull/471))
- [x] [TensorRT runtime](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) to improve inference speed ([#475](https://github.com/paninski-lab/lightning-pose/pull/475))
- [x] [pynvvideocodec](https://developer.nvidia.com/pynvvideocodec) to improve inference speed ([#477](https://github.com/paninski-lab/lightning-pose/pull/477))
- [x] add OpenCV video reader option for inference ([#491](https://github.com/paninski-lab/lightning-pose/pull/491))

## Losses and backbones
- [x] incorporate transformer backbones ([#84](https://github.com/paninski-lab/lightning-pose/pull/84), [#106](https://github.com/paninski-lab/lightning-pose/pull/106))
- [ ] compute non-temporal unsupervised losses on labeled data

## Multi-view support for non-mirrored setups
- [x] unsupervised losses for multi-view ([#187](https://github.com/paninski-lab/lightning-pose/pull/187))
- [x] context frames for multi-view ([#126](https://github.com/paninski-lab/lightning-pose/pull/126))
- [x] implement supervised datasets/dataloaders that work with multiple views ([#115](https://github.com/paninski-lab/lightning-pose/pull/115))

## Single-view dynamic crop (small animals in large frames)
- [x] context frames for dynamic crop ([#250](https://github.com/paninski-lab/lightning-pose/pull/250))
- [x] unsupervised losses for dynamic crop ([#250](https://github.com/paninski-lab/lightning-pose/pull/250))
- [x] implement dynamic cropping pipeline with detector model and pose estimator ([#250](https://github.com/paninski-lab/lightning-pose/pull/250))
- [x] split CLI `crop` command into `create_bbox` and `smooth_bbox` commands for modularity ([#420](https://github.com/paninski-lab/lightning-pose/pull/420))

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
