"""Extract frames to label from a list of videos."""

import argparse
import numpy as np
import os

from lightning_pose.utils.frame_selection import select_frame_idxs, export_frames


def run():

    args = parser.parse_args()

    for video_file in args.video_files:

        print(f"============== extracting frames from {video_file} ================")
        print("does this video file exist? %s" % ("YES" if os.path.exists(video_file) else "NO"))

        idxs_selected = select_frame_idxs(
            video_file=video_file, resize_dims=64, n_clusters=args.n_frames_per_video)

        video_name = os.path.splitext(os.path.basename(video_file))[0]
        save_dir = os.path.join(args.data_dir, video_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n_digits = 8
        format = "png"

        # save csv file inside same output directory
        if args.export_idxs_as_csv:
            frames_to_label = np.array([
                "img%s.%s" % (str(idx).zfill(n_digits), format) for idx in idxs_selected])
            np.savetxt(
                os.path.join(save_dir, "selected_frames.csv"),
                np.sort(frames_to_label),
                delimiter=',',
                fmt="%s")

        export_frames(
            video_file=video_file, save_dir=save_dir, frame_idxs=idxs_selected, format=format,
            n_digits=n_digits, context_frames=args.context_frames)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_files", action='append', default=[])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--n_frames_per_video", type=int, default=20)
    parser.add_argument("--context_frames", type=int, default=0)
    parser.add_argument("--export_idxs_as_csv", action="store_true")

    run()
