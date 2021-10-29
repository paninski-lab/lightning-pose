import fiftyone as fo
import pandas as pd
import os
import fiftyone.utils.annotations as foua
from tqdm import tqdm


# TODO: as param
video_path = "/home/jovyan/lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid.mp4"

"""video_path, if mp4, should be codec h.264. if not:
video_path_transformed = "/home/jovyan/lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/transformed_test_vid.mp4"
import fiftyone.utils.video as fouv
fouv.reencode_video(video_path, video_path_transformed, verbose=False)
"""

assert os.path.isfile(video_path)

video_sample = fo.Sample(filepath=video_path)

# TODO: as param
csv_with_preds = pd.read_csv(
    "/home/jovyan/lightning-pose/toy_datasets/toymouseRunningData/unlabeled_videos/test_vid_heatmap.csv",
    header=[1, 2],
)

# TODO: make as params
height = 406
width = 396

# Example from fiftyone team
# fo.Keypoints(keypoints=[fo.Keypoint(points=[[0.5, 0.5]], confidence=0.5), ...])
# TODO: check how to populate and use the confidence field

keypoint_names = csv_with_preds.columns.levels[0][1:]
print("Populating the per-frame keypoints...")
for frame_idx in tqdm(range(csv_with_preds.shape[0])):  # loop over frames
    keypoints_list = []
    for kp_name in keypoint_names:  # loop over bp names
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
    video_sample.frames[frame_idx + 1]["preds"] = fo.Keypoints(keypoints=keypoints_list)
print("Done.")

# TODO: name as param
dataset = fo.Dataset()
dataset.add_sample(video_sample)
dataset.compute_metadata()
# print(dataset)

# TODO: control more params from outside. also for app
config = foua.DrawConfig(
    {"keypoints_size": 9}
)  # note that 9 is approximately 40+ times smaller than the image

# launch an interactive session
session = fo.launch_app(dataset, remote=True)
session.wait()

# TODO: by definition always save to disk.
config = foua.DrawConfig({"keypoints_size": 9})
outpath = "/home/jovyan/vid.mp4"
print("Writing labeled images to '%s'" % outpath)
foua.draw_labeled_video(video_sample, outpath, config=config)
print("Writing complete")

# save to disc

# inspect frames
# for frame_number, frame in video_sample.frames.items():
#     print(frame)
#     break

# bodypart_names = csv_with_preds.columns.levels[0][1:]
# print("Populating the per-frame keypoints...")
# for frame_idx in tqdm(range(csv_with_preds.shape[0])):  # loop over frames
#     keypoints_list = []
#     for bp in bodypart_names:  # loop over bp names
#         # that's quick but looses body part name labels. works
#         keypoints_list.append(
#             (
#                 csv_with_preds[bp]["x"][frame_idx] / width,
#                 csv_with_preds[bp]["y"][frame_idx] / height,
#             )
#         )
#     # # the below used to work note that their indexing starts from 1,
#     video_sample.frames[frame_idx + 1]["preds"] = fo.Keypoints(
#         keypoints=[fo.Keypoint(points=keypoints_list)]
#     )
# print("Done.")
