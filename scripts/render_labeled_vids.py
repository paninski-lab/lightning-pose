import fiftyone as fo
import pandas as pd
import os
import fiftyone.utils.annotations as foua
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from pose_est_nets.utils.io import verify_absolute_path
from pose_est_nets.utils.plotting_utils import get_videos_in_dir
import warnings


"""For FifyOne visualization, videos, if mp4, should be codec h.264. If they're not, see below on how to convert them:
video_path_transformed = "/home/jovyan/lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/transformed_test_vid.mp4"
import fiftyone.utils.video as fouv
fouv.reencode_video(video_path, video_path_transformed, verbose=False)
"""


def make_keypoint_list(
    csv_with_preds, keypoint_names: list, frame_idx: int, width: int, height: int
) -> list:
    keypoints_list = []
    for kp_name in keypoint_names:  # loop over names
        # write a single keypoint's position, confidence, and name
        keypoints_list.append(
            fo.Keypoint(
                points=[
                    [
                        csv_with_preds[kp_name]["x"][frame_idx] / width,
                        csv_with_preds[kp_name]["y"][frame_idx] / height,
                    ]
                ],
                confidence=csv_with_preds[kp_name]["likelihood"][frame_idx],
                label=kp_name,
            )
        )
    return keypoints_list


@hydra.main(config_path="configs", config_name="config")
def render_labeled_videos(cfg: DictConfig):
    """This function currently supports a single video. It takes the .csv including predictions from one or more models.
    TODO: we have to decide together how to generalize to multiple videos, without inputting all the csv's for each model.

    Args:
        cfg (DictConfig): our hierarchical Hydra config. of special interest is cfg.eval.
    """
    # there may be multiple video paths in the cfg eval
    # certainly there may be multiple models.
    # loop over both
    video_dir = verify_absolute_path(
        cfg.eval.path_to_test_videos[0]
    )  # TODO: just for now assuming there's one dir, but can have multiple dirs.
    video = get_videos_in_dir(video_dir)
    if len(video) > 1:  # just until further support
        warnings.warn(
            "{} has more than one video. \nAnalyzing just {}".format(
                video_dir, video[0]
            )
        )
    # currently supporting a single video in a directory, TODO: extend and loop
    dataset = fo.Dataset(
        cfg.eval.fifty_one_dataset_name + "_videos"
    )  # TODO: in the future, dataset should include multiple video samples
    video_sample = fo.Sample(filepath=video[0])  # TODO: again extend to multiple vids
    for display_name, path_to_preds_file in zip(
        cfg.eval.model_display_names, cfg.eval.path_to_csv_predictions
    ):  # loop over different models' preds for a given vid
        print(
            "========= \nModel display name: {} \nPredictions file: {}  \nVideo analyzed: {}".format(
                display_name, path_to_preds_file, video[0]
            )
        )
        absolute_path_to_preds_file = verify_absolute_path(
            path_to_preds_file
        )  # if toy_dataset, append absolute path. else do nothing.
        df_with_preds = pd.read_csv(absolute_path_to_preds_file, header=[1, 2])
        keypoint_names = df_with_preds.columns.levels[0][1:]
        print("Populating the per-frame keypoints...")
        for frame_idx in tqdm(range(df_with_preds.shape[0])):  # loop over frames
            keypoints_list = make_keypoint_list(
                csv_with_preds=df_with_preds,
                keypoint_names=keypoint_names,
                frame_idx=frame_idx,
                width=cfg.data.image_orig_dims.width,
                height=cfg.data.image_orig_dims.height,
            )
            video_sample.frames[frame_idx + 1][display_name] = fo.Keypoints(
                keypoints=keypoints_list
            )

    dataset.add_sample(video_sample)
    dataset.compute_metadata()

    # TODO: control more params from outside?
    # After you finish interactively looking at the video it, save to disc
    config = foua.DrawConfig(
        {"keypoints_size": 9}
    )  # this size is good for a 400X400 image.
    outpath = (
        video[0].replace(".mp4", "") + "_labeled.mp4"
    )  # TODO: careful with [0], yoy should loop over videos.
    print("Writing a labeled video to '%s'" % outpath)
    foua.draw_labeled_video(video_sample, outpath, config=config)
    print("Video writing now complete.")

    # launch an interactive session
    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == "__main__":
    render_labeled_videos()
