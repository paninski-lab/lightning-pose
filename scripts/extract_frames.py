"""Extract frames to label from a list of videos."""

import argparse

from lightning_pose.utils.frame_selection import select_frame_idxs, export_frames


def run():

    for video_file in args.video_files:

        idxs_selected = select_frame_idxs(
            video_file=video_file, resize_dims=64, n_clusters=args.n_frames_per_video)

        save_dir = os.path.join(
            args.data_dir,
            os.path.splitext(os.path.basename(video_file))[0]
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        export_frames(
            video_file=video_file, save_dir=save_dir, frame_idxs=idxs_selected, format="png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_files", action='append', default=[])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--n_frames_per_video", type=int)

    run()
