import os
import shutil
import argparse
import numpy as np

"""
python lp2eks.py --input_dir /path/to/input --output_dir /path/to/output
"""


def process_folders(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all directories in the input folder
    dirs = os.listdir(input_dir)
    
    # Copy and rename files
    for idx, dir in enumerate(dirs):
        dirpath = os.path.join(input_dir, dir)
        predpath = os.path.join(dirpath, 'video_preds')

        if os.path.exists(predpath):  # Ensure the path exists
            for file in os.listdir(predpath):
                if file.endswith('Cam2.csv'):
                    source_file = os.path.join(predpath, file)
                    new_name = file.replace('.csv', f".rng={idx}.csv")
                    destination_file = os.path.join(output_dir, new_name)

                    shutil.copy(source_file, destination_file)
    
    # Create subdirectories in the output folder based on unique prefixes
    dirs = np.unique([x.split('.rng')[0] for x in os.listdir(output_dir)])
    for x in dirs:
        os.mkdir(os.path.join(output_dir, x))
    
    # Move files into their respective subdirectories
    for file in os.listdir(output_dir):
        if file.endswith('csv'):
            old_path = os.path.join(output_dir, file)
            new_folder = os.path.join(output_dir, file.split('.rng')[0])
            new_path = os.path.join(new_folder, file)
            os.rename(old_path, new_path)

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Organize video predictions for EKS.")
    parser.add_argument("--input_dir", required=True, help="Path to the input folder containing subdirectories.")
    parser.add_argument("--output_dir", default="for_eks", help="Path to the output directory. Defaults to 'for_eks'.")

    args = parser.parse_args()
    
    # Process the folders
    process_folders(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
