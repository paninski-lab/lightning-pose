import argparse
from lightning_pose.api.model import Model

parser = argparse.ArgumentParser(description="Process videos using a model.")
parser.add_argument("model_dir", help="Path to the model directory.")
parser.add_argument("videos", nargs="+", help="List of video file paths.")

args = parser.parse_args()

model = Model.from_dir(args.model_dir)
model.predict_on_video_file_multiview(args.videos, generate_labeled_video=True)
