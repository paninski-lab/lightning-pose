import os
import argparse
import pandas as pd

"""
python lp-eks2simba.py --input_dir /path/to/input --output_dir /path/to/output
"""

def process_files(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            # Process each CSV file
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path)
            
            delimiter = "_"
            combined = [f"{a}{delimiter}{b}" for a, b in zip(df.iloc[0, :], df.iloc[1, :])]
            df.iloc[0, :] = combined
            df = df.iloc[0:, 1:]
            
            df.columns = df.iloc[0]
            df = df[1:]
            df.reset_index(drop=True, inplace=True)
            df = df.iloc[1:, :]
            df.columns = [x.replace('likelihood', 'p') for x in df.columns]
            
            # Filter columns
            df = df[[x for x in df.columns if len(x.split('_')) < 4]]
            
            # Save the processed file to the output directory
            output_file = os.path.join(output_dir, file.split('_eks_singlecam')[0] + '.csv')
            df.to_csv(output_file, index=False)

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Process CSV files for Simba.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing CSV files.")
    parser.add_argument("--output_dir", default=None, help="Path to the output directory. Defaults to input_dir/for_simba.")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "for_simba")
    
    process_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
