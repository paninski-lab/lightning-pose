import os 
from typing import List
import subprocess


def find_vids_in_dir(folder_path: str) -> List[str]:
    # get all video files in directory
    vid_files = [f for f in os.listdir(folder_path) if f.endswith((".mp4", ".avi", ".mov"))]
    # get absolute paths of video files and check that they exist
    absolute_paths = [os.path.join(folder_path, v) for v in vid_files if os.path.isfile(os.path.join(folder_path, v))]
    return absolute_paths

import subprocess

def reencode_video(input_file: str, output_file: str) -> None:
    """ a function that executes ffmpeg from a subprocess
    reencodes video into H.264 coded format
    input_file: str with abspath to existing video
    outputfile: str with abspath to to new video"""
    assert os.path.isfile(input_file), "input video does not exist." # input file exists
    assert os.path.isdir(os.path.dirname(output_file)), "saving folder %s does not exist." % os.path.dirname(output_file) # folder for saving outputs exists
    ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -c:a copy -y {output_file}'
    subprocess.run(ffmpeg_cmd, shell=True)

def check_codec_format(input_file: str):
    # Run FFprobe command to get video codec version
    # command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_version", "-of", "default=noprint_wrappers=1:nokey=1", input_file]
    ffmpeg_cmd = f'ffmpeg -i {input_file}'
    output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    output_str = output_str.stderr # stderr because the ffmpeg command has no output file. but the stderr still has codec info.

    # search for h264
    if output_str.find('h264') != -1:
        # print('Video uses H.264 codec')
        is_codec = True
    else:
        # print('Video does not use H.264 codec')
        is_codec = False
    return is_codec



