# Consolidated studio research baseline — 2026-09-17

Active branch: post_sub_lp (paired with mouse-pose post_sub_mm). Renamed from
research/zero-shot-baseline on 2026-09-18 without changing the baseline code.
The original super_mouse_paper branch
and all historical experiment checkouts remain available.

Integrated: optimizer-step scheduling fix (source0a1e8ab, cherry-picked as73b169d), preserving
explicit legacy epoch milestones. Optional training.imgaug_seed makes the research recipe's
augmentation seed explicit without changing existing config behavior. New models trained
with corrected schedules must not resume old epoch-scheduler optimizer states as step states.
Exact sampler/augmentation stream resume remains a separate unresolved feature.

Not integrated: experimental occlusion loss weights or clean-teacher/paired-view anchor code.
Existing anchored LoRA behavior is unchanged. Existing multiview patch masking is not enabled
for single-view heatmap models. DINOv3 ViT-B support already exists and needs no architecture
change here; verified research recipes live in ../mouse-pose/configs/zero_shot/.

See ../mouse-pose/docs/zero_shot_baseline.md for evidence, metrics, run procedure and rollback.
Historical teachers remain intact; improved all-data scores are not evidence of zero-shot gains.
