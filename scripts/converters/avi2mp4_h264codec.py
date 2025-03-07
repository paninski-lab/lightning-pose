import os
import subprocess
import argparse

def reencode_video(input_file: str, output_file: str) -> None:
    """Reencodes video into H.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: Absolute path to the existing video.
        output_file: Absolute path to the new mp4 video using H.264 codec.
    """
    assert os.path.isfile(input_file), f"Input video does not exist: {input_file}"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ffmpeg_cmd = f'ffmpeg -i "{input_file}" -c:v libx264 -pix_fmt yuv420p -c:a copy -y "{output_file}"'
    subprocess.run(ffmpeg_cmd, shell=True, check=True)

    print(f"Converted: {input_file} -> {output_file}")


def batch_reencode_videos(input_dir: str, output_dir: str) -> None:
    """Batch processes all .avi files in the input directory to .mp4 in the output directory.

    Args:
        input_dir: Directory containing .avi files.
        output_dir: Directory to save the converted .mp4 files.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.endswith(".avi")]
    if not video_files:
        print(f"No .avi files found in directory: {input_dir}")
        return

    for video_file in video_files:
        input_file = os.path.join(input_dir, video_file)
        output_file = os.path.join(output_dir, video_file.replace(".avi", ".mp4"))
        try:
            reencode_video(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    print("Batch processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert .avi videos to .mp4 with H.264 codec.")
    parser.add_argument("input_dir", type=str, help="Directory containing input .avi files.")
    parser.add_argument("output_dir", type=str, help="Directory to save output .mp4 files.")
    args = parser.parse_args()

    batch_reencode_videos(args.input_dir, args.output_dir)
