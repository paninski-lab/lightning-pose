import argparse
import warnings

from lightning_pose.api.model import Model


# Issue deprecation warning
warnings.warn(
    "This script is deprecated and will be removed in a future version. "
    "Please use the command line interface instead:\n\n"
    "To predict on a folder of video files:\n"
    "  litpose predict <model_dir> <video_files_dir>\n\n"
    "For more information, visit:\n"
    "https://lightning-pose.readthedocs.io/en/latest/source/user_guide_multiview/training_inference.html",
    DeprecationWarning,
    stacklevel=2
)


parser = argparse.ArgumentParser(description="Process videos using a model.")
parser.add_argument("model_dir", help="Path to the model directory.")
parser.add_argument("videos", nargs="+", help="List of video file paths.")

args = parser.parse_args()

model = Model.from_dir(args.model_dir)
model.predict_on_video_file_multiview(args.videos, generate_labeled_video=True)
