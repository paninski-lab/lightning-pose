"""Test the FiftyOne module."""

import os
import pandas as pd
import shutil


def test_fiftyone_image_plotter(cfg, tmpdir):

    from lightning_pose.utils.fiftyone import FiftyOneImagePlotter

    # copy ground truth labels to a "predictions" file
    data_dir_abs = os.path.abspath(cfg.data.data_dir)
    gt_file = os.path.join(data_dir_abs, cfg.data.csv_file)
    model_dir = os.path.join(tmpdir, "date", "time")
    os.makedirs(model_dir, exist_ok=True)
    pred_file = os.path.join(model_dir, "predictions.csv")
    shutil.copyfile(gt_file, pred_file)

    # add final column to "predictions"
    df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
    df.loc[:, ("set", "", "")] = "train"
    df.to_csv(pred_file)
    n_preds = df.shape[0]

    # update config
    cfg_new = cfg.copy()
    cfg_new.data.data_dir = data_dir_abs
    cfg_new.eval.model_display_names = ["test_model"]
    cfg_new.eval.hydra_paths = [model_dir]
    cfg_new.eval.fiftyone.dataset_name = str(tmpdir)  # get unique dataset name

    # make fiftyone dataset and check
    plotter = FiftyOneImagePlotter(cfg=cfg_new)
    dataset = plotter.create_dataset()

    assert len(dataset) == n_preds
