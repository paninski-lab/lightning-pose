import torch
import sklearn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def video_pipe(
    filenames: list,
    resize_dims: Optional[list],
    random_shuffle: Optional[bool] = False,
):  # TODO: what does it return? more typechecking
    video = fn.readers.video(
        device=_DALI_DEVICE,
        filenames=filenames,
        sequence_length=_SEQUENCE_LENGTH_UNSUPERVISED,
        random_shuffle=random_shuffle,
        initial_fill=_INITIAL_PREFETCH_SIZE,
        normalized=False,
        dtype=types.DALIDataType.FLOAT,
    )
    video = fn.resize(video, size=resize_dims)
    video = (
        video / 255.0
    )  # original videos (at least Rick's) range from 0-255. transform it to 0,1. # TODO: not sure that we need that, make sure it's the same as the supervised ones
    transform = fn.crop_mirror_normalize(
        video,
        output_layout="FCHW",
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
    )
    return transform

 class LightningWrapper(DALIGenericIterator):
	def __init__(self, *kargs, **kvargs):
	    super().__init__(*kargs, **kvargs)

	def __len__(self):  # just to avoid ptl err check
	    return 1  # num frames = len * batch_size; TODO: determine actual length of vid

	def __next__(self):
	    out = super().__next__()
	    return torch.tensor(
	        out[0]["x"][
	            0, :, :, :, :
	        ],  # should be batch_size, W, H, 3. TODO: valid for one sequence.
	        dtype=torch.float,  # , device="cuda"
	    )


 @typechecked
def PCA_prints(pca: sklearn.decomposition._pca.PCA, components_to_keep: int) -> None:
    print("Results of running PCA on labels:")
    print(
        "explained_variance_ratio_: {}".format(
            np.round(pca.explained_variance_ratio_, 3)
        )
    )
    print(
        "total_explained_var: {}".format(
            np.round(np.sum(pca.explained_variance_ratio_[:components_to_keep]), 3)
        )
    )

