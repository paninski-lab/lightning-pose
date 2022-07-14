python scripts/train_hydra.py --multirun \
training.max_epochs=11 \
training.train_batch_size=8 \
training.train_frames=125 \
model.model_name="singleview_sweep" \
