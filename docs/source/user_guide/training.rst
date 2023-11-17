###################
Training
###################

To train a model on the example dataset provided with this repo, run the following command:
```console
python scripts/train_hydra.py
```

To train a model on your own dataset, follow these steps:
1. ensure your data is in the [proper data format](docs/directory_structures.md)
2. copy the file `scripts/configs/config_default.yaml` to another directory and rename it.
You will then need to update the various fields to match your dataset (such as image height and width).
See other config files in `scripts/configs/` for examples.
3. train your model from the terminal and overwrite the config path and config name with your newly created file:

```console
python scripts/train_hydra.py --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" --config-name="<CONFIG_NAME.yaml>"
```

You can find more information on the structure of the model directories
[here](docs/directory_structures.md#model-directory-structure).

## Working with `hydra`

For all of the scripts in our `scripts` folder, we rely on `hydra` to manage arguments in
config files. You have two options: directly edit the config file, or override it from the command
line.

- **Edit** the hydra config, that is, any of the parameters in, e.g.,
`scripts/configs/config_mirror-mouse-example.yaml`, and save it.
Then run the script without arguments:
```console
python scripts/train_hydra.py
```

- **Override** the argument from the command line; for example, if you want to use a maximum of 11
  epochs instead of the default number (not recommended):
```console
python scripts/train_hydra.py training.max_epochs=11
```

Or, for your own dataset,
```console
python scripts/train_hydra.py --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" --config-name="<CONFIG_NAME.yaml> training.max_epochs=11
```

We also recommend trying out training with automatic resizing to smaller images first;
this allows for larger batch sizes/fewer Out Of Memory errors on the GPU:
```console
python scripts/train_hydra.py --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" --config-name="<CONFIG_NAME.yaml> data.image_resize_dims.height=256 data.image_resize_dims.width=256
```

See more documentation on the config file fields [here](docs/config.md).

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs,
will be saved at the `lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS/tb_logs` directory. (Note: this
behavior can be changed by updating `hydra.run.dir` in the config yaml to an absolute path of your
choosing.)

To view the logged losses with tensorboard in your browser, in the command line, run:
```console
tensorboard --logdir outputs/YYYY-MM-DD/
```
where you use the date in which you ran the model.
Click on the provided link in the terminal, which will look something like `http://localhost:6006/`.
Note that if you save the model at a different directory, just use that directory after `--logdir`.
