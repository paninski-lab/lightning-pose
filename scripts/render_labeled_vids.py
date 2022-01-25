"""Overlay model predictions on videos with fiftyone.

Videos, if mp4 format, should be codec h.264. If they're not, convert them as follows:

```python
import fiftyone.utils.video as fouv
video_path = /path/to/video.mp4
video_path_transformed = "/path/to/video_transformed.mp4"
fouv.reencode_video(video_path, video_path_transformed, verbose=False)
```

"""

import fiftyone as fo
import fiftyone.utils.annotations as foua
import hydra
from omegaconf import DictConfig, OmegaConf

from sympy import pretty_print
from lightning_pose.utils.io import return_absolute_path
from lightning_pose.utils.plotting_utils import get_videos_in_dir
from lightning_pose.utils.fiftyone import FiftyOneKeypointVideoPlotter, check_dataset
from lightning_pose.utils.scripts import pretty_print_str


@hydra.main(config_path="configs", config_name="config")
def render_labeled_videos(cfg: DictConfig):
    """Overlay keypoints from already saved prediction csv file on video.

    This function currently supports a single video. It takes the prediction csv from
    one or more models.

    TODO: how to generalize to multiple videos, without inputting csv for each model.

    Args:
        cfg (DictConfig): hierarchical Hydra config. of special interest is cfg.eval

    """

    fo_video_class = FiftyOneKeypointVideoPlotter(cfg=cfg)
    dataset = fo_video_class.create_dataset()
    check_dataset(dataset)

    config = foua.DrawConfig(
        {
            "keypoints_size": 9,
            "show_keypoints_names": False,
            "show_keypoints_labels": False,
            "show_keypoints_attr_names": False,
            "per_keypoints_label_colors": False,
        }
    )  # this size is good for a 400X400 image.

    outpath = fo_video_class.video.replace(".mp4", "") + "_labeled.mp4"
    pretty_print_str("Writing a labeled video to '%s'..." % outpath)
    foua.draw_labeled_video(dataset[fo_video_class.video], outpath, config=config)
    pretty_print_str("Video writing now complete!")

    # launch an interactive session
    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == "__main__":
    render_labeled_videos()
