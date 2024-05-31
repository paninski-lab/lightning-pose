"""Test the FiftyOne module."""
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd


def test_fiftyone_image_plotter(cfg, tmpdir):

    from lightning_pose.utils.fiftyone import FiftyOneImagePlotter

    # copy ground truth labels to a "predictions" file
    data_dir_abs = os.path.abspath(cfg.data.data_dir)
    gt_file = os.path.join(data_dir_abs, cfg.data.csv_file)
    model_name = datetime.today().strftime("%Y-%m-%d/%H-%M-%S_PYTEST")
    model_dir = os.path.join(tmpdir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    pred_file = os.path.join(model_dir, "predictions.csv")
    shutil.copyfile(gt_file, pred_file)

    # add final column to "predictions"
    df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
    df.loc[:, ("set", "", "")] = "train"
    df.to_csv(pred_file)
    n_preds = df.shape[0]

    # -----------------------
    # test 1: standard setup
    # -----------------------
    # update config
    cfg_new = cfg.copy()
    cfg_new.data.data_dir = data_dir_abs
    cfg_new.eval.model_display_names = ["test_model"]
    cfg_new.eval.hydra_paths = [model_dir]
    cfg_new.eval.fiftyone.dataset_name = str(model_dir)  # get unique dataset name

    # make fiftyone dataset and check
    plotter = FiftyOneImagePlotter(cfg=cfg_new)
    dataset = plotter.create_dataset()

    assert len(dataset) == n_preds

    # -----------------------
    # test 2: missing labels
    # -----------------------
    # load ground truth file
    gt_df = pd.read_csv(gt_file, header=[0, 1, 2], index_col=0)
    # set all ground truth labels to nan
    gt_df.iloc[:, :] = np.nan
    csv_file = os.path.join(tmpdir, "CollectedData_all_nans")
    gt_df.to_csv(csv_file)

    # update config
    cfg_new = cfg.copy()
    cfg_new.data.data_dir = data_dir_abs
    cfg_new.data.csv_file = csv_file
    cfg_new.eval.model_display_names = ["test_model"]
    cfg_new.eval.hydra_paths = [model_dir]
    cfg_new.eval.fiftyone.dataset_name = str(model_dir) + "1"  # get unique dataset name

    # make fiftyone dataset and check
    plotter = FiftyOneImagePlotter(cfg=cfg_new)
    dataset = plotter.create_dataset()

    assert len(dataset) == n_preds
