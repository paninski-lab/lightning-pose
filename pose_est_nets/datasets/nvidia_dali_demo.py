from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
import os
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "data/images"
max_batch_size = 8


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_dir = "~/mouseRunningData/barObstacleScaling1"
max_batch_size = 8

@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels

pipe = simple_pipeline()
pipe.build()

@pipeline_def
def video_pipe(filenames):  # TODO: review, fix means
    initial_prefetch_size = 16
    video = fn.readers.video(
        device="gpu",  # TODO: check what needs to be device for tests to run on CPU
        filenames=filenames,
        sequence_length=1,
        initial_fill=initial_prefetch_size,
        normalized=False,
        dtype=types.DALIDataType.FLOAT,
    )
    video = video / 255
    transform = fn.crop_mirror_normalize(
        video,
        output_layout="FCHW",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

# Testing the video_pipe
video_path = "~/mouseRunningData/unlabeled_videos/180727_001.mp4"
data_pipe = video_pipe([video_path])  # TODO: called with wrong args below, verify
data_pipe.build()
        for i in range(startup_len):
            data_pipe.run()
        print(data_pipe._api_type)
        data_pipe._api_type = types.PipelineAPIType.ITERATOR
        print(data_pipe._api_type)
        self.semi_supervised_loader = LightningWrapper(  # TODO: why write to self?
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )  # changed output_map to account for dummy labels
# Testing the video_pipe