import os 
from lightning_pose.utils.video_preprocessing import find_vids_in_dir, reencode_video, check_codec_format
import argparse 
import shutil

# Use the shutil.copy() method to copy the file from the source to the destination



# argparse boilerplate
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder_path', type=str, help='Path to the folder')
args = parser.parse_args()

# Access the folder_path argument
folder_path = args.folder_path

if __name__ == "__main__":
    assert os.path.isdir(args.folder_path), f"Video folder {args.folder_path} does not exist."
    print("=================================")
    print(f"Checking video folder {args.folder_path}...")
    print("=================================")
    # find .mov/.avi/.mp4 videos in directory
    videos_list = find_vids_in_dir(args.folder_path)
    # make new dir for reencoded vids
    new_dir_name = args.folder_path + '_reencoded'
    os.makedirs(new_dir_name, exist_ok=True)
    
    # loop over them and reencode
    for input_file in videos_list:
        is_mp4 = input_file.endswith('.mp4') # is .mp4?
        is_codec = check_codec_format(input_file=input_file) # is codec h.264?
        basename = os.path.basename(input_file)
        output_file = os.path.join(new_dir_name, basename)

        if not is_mp4:
            # change output_file to have mp4
            split_name = output_file.split(".")[:-1]
            split_name.append("mp4")
            output_file = ".".join(split_name)
         
        print(f'Saving input {input_file} to {output_file}')
        if (is_codec == False) or (is_mp4 == False): # if either of these isn't true
            # reencode and save in a new dir
            print("Reencoding using FFMPEG...")
            reencode_video(input_file=input_file, output_file=output_file)
            print("Done!")

        else: # vid is fine, just copying to a new dir
            print("Copying video as is...")
            shutil.copy(input_file, output_file)
            print("Done!")
