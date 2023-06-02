import cv2
import os
import numpy as np
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import subprocess
from typeguard import typechecked
from typing import List


from lightning_pose.data.dali import video_pipe, LitDaliWrapper
from lightning_pose.data.utils import count_frames


@typechecked
def select_frame_idxs(video_file: str, resize_dims: int = 64, n_clusters: int = 20) -> np.ndarray:
    """Cluster a low-dimensional representation of high motion energy frames.

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    resize_dims: height AND width of resizing operation to make PCA faster
    n_clusters: total number of frames to return

    Returns
    -------
    array-like

    """

    seq_len = 128

    # ---------------------------------------------------------
    # read video with DALI
    # ---------------------------------------------------------
    # set up parameters for DALI video pipe
    pipe_args = {
        "filenames": video_file,
        "resize_dims": [resize_dims, resize_dims],
        "sequence_length": seq_len,
        "step": seq_len,
        "batch_size": 1,
        "num_threads": 4,
        "device_id": 0,
        "random_shuffle": False,
        "device": "gpu",
        "name": "reader",
        "pad_sequences": True,
    }
    pipe = video_pipe(**pipe_args)

    # set up parameters for pytorch iterator
    frame_count = count_frames(video_file)
    num_iters = int(np.ceil(frame_count / pipe_args["sequence_length"]))
    iterator_args = {
        "num_iters": num_iters,
        "eval_mode": "predict",
        "do_context": False,
        "output_map": ["frames", "transforms"],
        "last_batch_policy": LastBatchPolicy.FILL,
        #     "last_batch_padded": True,
        "auto_reset": False,
        "reader_name": "reader",
    }

    # build iterator
    iterator = LitDaliWrapper(pipe, **iterator_args)

    # collect all data
    batches = []
    for batch in tqdm(iterator):
        # batches.append(batch)
        # take mean over color channel, remove spatial dims
        # result is shape (batch_size, height * width)
        batches.append(
            torch.reshape(
                torch.mean(batch["frames"], dim=1), (batch["frames"].shape[0], -1)
            )
            .detach()
            .cpu()
            .numpy().astype(np.float16)  # reduce memory overhead
        )
    batches = np.concatenate(batches, axis=0)[:(frame_count - 2)]  # leave room for context

    # ---------------------------------------------------------
    # get example frames by using kmeans in pc space (high me)
    # ---------------------------------------------------------
    # take temporal diffs
    print('computing motion energy...')
    me = np.concatenate([
        np.zeros((1, batches.shape[1])).astype(np.float16),
        np.diff(batches, axis=0)
    ])
    # take absolute values and sum over all pixels to get motion energy
    me = np.sum(np.abs(me), axis=1)

    # find high me frames, defined as those with me larger than nth percentile me
    prctile = 50 if frame_count < 1e5 else 75  # take fewer frames if there are many
    idxs_high_me = np.where(me > np.percentile(me, prctile))[0]

    # compute pca over high me frames
    print('performing pca over high motion energy frames...')
    pca_obj = PCA(n_components=np.min([batches[idxs_high_me].shape[0], 32]))
    embedding = pca_obj.fit_transform(X=batches[idxs_high_me])
    del batches  # free up memory

    # cluster low-d pca embeddings
    print('performing kmeans clustering...')
    kmeans_obj = KMeans(n_clusters=n_clusters, n_init="auto")
    kmeans_obj.fit(X=embedding)

    # find high me frame that is closest to each cluster center
    # kmeans_obj.cluster_centers_ is shape (n_clusters, n_pcs)
    centers = kmeans_obj.cluster_centers_.T[None, ...]
    # embedding is shape (n_frames, n_pcs)
    dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
    # dists is shape (n_frames, n_clusters)
    idxs_prototypes_ = np.argmin(dists, axis=0)
    # now index into high me frames to get overall indices
    idxs_prototypes = idxs_high_me[idxs_prototypes_]

    # free up gpu
    del batch
    del pipe
    del iterator
    torch.cuda.empty_cache()

    return idxs_prototypes


@typechecked
def export_frames(
    video_file: str,
    save_dir: str,
    frame_idxs: np.ndarray,
    format: str = "png",
    n_digits: int = 8,
    context_frames: int = 0,
):
    """

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    save_dir: absolute path to directory in which selected frames are saved
    frame_idxs: indices of frames to grab
    format: only "png" currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save

    """

    cap = cv2.VideoCapture(video_file)

    # expand frame_idxs to include context frames
    if context_frames > 0:
        context_vec = np.arange(-context_frames, context_frames + 1)
        frame_idxs = (frame_idxs.squeeze()[None, :] + context_vec[:, None]).flatten()
        frame_idxs.sort()
        frame_idxs = frame_idxs[frame_idxs >= 0]
        frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_idxs = np.unique(frame_idxs)

    # load frames from video
    frames = get_frames_from_idxs(cap, frame_idxs)

    # save out frames
    os.makedirs(save_dir, exist_ok=True)
    for frame, idx in zip(frames, frame_idxs):
        cv2.imwrite(
            filename=os.path.join(save_dir, "img%s.%s" % (str(idx).zfill(n_digits), format)),
            img=frame[0],
        )


def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype="uint8")
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                "warning! reached end of video; returning blank frames for remainder of "
                + "requested indices"
            )
            break
    return frames


@typechecked
def reencode_video(input_file: str, output_file: str) -> None:
    """reencodes video into H.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: abspath to existing video
        output_file: abspath to to new video

    """

    # check input file exists
    assert os.path.isfile(input_file), "input video does not exist."
    # check directory for saving outputs exists
    assert os.path.isdir(
        os.path.dirname(output_file)), \
        f"saving folder {os.path.dirname(output_file)} does not exist."
    ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -pix_fmt yuv420p -c:a copy -y {output_file}'
    subprocess.run(ffmpeg_cmd, shell=True)


@typechecked
def check_codec_format(input_file: str):
    """Run FFprobe command to get video codec and pixel format."""

    ffmpeg_cmd = f'ffmpeg -i {input_file}'
    output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    # stderr because the ffmpeg command has no output file, but the stderr still has codec info.
    output_str = output_str.stderr

    # search for correct codec (h264) and pixel format (yuv420p)
    if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
        # print('Video uses H.264 codec')
        is_codec = True
    else:
        # print('Video does not use H.264 codec')
        is_codec = False
    return is_codec
