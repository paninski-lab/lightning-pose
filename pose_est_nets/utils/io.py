from operator import sub
import os
import pickle
from typeguard import typechecked
from typing import Any


@typechecked
def set_or_open_folder(folder_path: str) -> str:
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("Opened a new folder at: {}".format(folder_path))
    else:
        print("The folder already exists at: {}".format(folder_path))
    return folder_path


# pickle utils
@typechecked
def save_object(obj: Any, filename: str) -> None:
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


@typechecked
def load_object(filename: str) -> Any:
    with open(filename, "rb") as input:  # note rb and not wb
        return pickle.load(input)


@typechecked
def get_latest_version(lightning_logs_path: str) -> str:
    # TODO: add what's needed to pull out ckpts
    subfolders = os.listdir(lightning_logs_path)
    if ".DS_Store" in subfolders:
        subfolders.remove(".DS_Store")
    # ints = []
    # for l in subfolders:
    #     print(l)
    #     if l != '.DS_Store':
    #         ints.append(int(l.split('_')[-1]))
    #     else:
    #         subfolders.remove(l)
    ints = [int(l.split("_")[-1]) for l in subfolders]
    latest_version = subfolders[ints.index(max(ints))]
    print("version used: %s" % latest_version)
    return latest_version


def get_absolute_hydra_path_from_hydra_str(hydra_path: str) -> str:
    import os

    cwd_split = os.getcwd().split(os.path.sep)
    desired_path_list = cwd_split[:-2]
    absolute_path = os.path.join(os.path.sep, *desired_path_list, hydra_path)
    assert os.path.isdir(absolute_path)
    return absolute_path


def ckpt_path_from_base_path(
    base_path: str,
    model_name: str,
    logging_dir_name: str = "tb_logs/",
    version: int = 0,
) -> str:
    """Given a path to a hydra output with trained model, extract the .ckpt to later load weights.

    Args:
        base_path (str): path to a folder with logs and checkpoint. for example, function will search base_path/logging_dir_name/model_name...
        model_name (str): the name you gave your model before training it; appears as model_name in lightning-pose/scripts/config/model_params.yaml
        logging_dir_name (str, optional): name of the folder in logs, controlled in train_hydra.py Defaults to "tb_logs/".

    Returns:
        str: path to model checkpoint
    """
    # TODO: consider finding the most recent hydra path containing logging_dir_name
    import glob
    import os

    model_search_path = os.path.join(
        base_path,
        logging_dir_name,  # TODO: may change when we switch from Tensorboard
        model_name,  # get the name string of the model (determined pre-training)
        "version_%i"
        % version,  # always version_0 because ptl starts a version_0 folder in a new hydra outputs folder
        "checkpoints",
        "*.ckpt",
    )

    # TODO: we're taking the last ckpt. make sure that with multiple checkpoints, this is what we want
    model_ckpt_path = glob.glob(model_search_path)[-1]
    return model_ckpt_path
