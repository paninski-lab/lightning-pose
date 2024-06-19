import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dlc_dir", type=str)
parser.add_argument("--lp_dir", type=str)
args = parser.parse_args()
dlc_dir = args.dlc_dir
lp_dir = args.lp_dir

print(f"Converting DLC project located at {dlc_dir} to LP project located at {lp_dir}")

# check provided DLC path exists
if not os.path.exists(dlc_dir):
    raise NotADirectoryError(f"did not find the directory {dlc_dir}")

# check paths are not the same
if dlc_dir == lp_dir:
    raise NameError(f"dlc_dir and lp_dir cannot be the same")

# find all labeled data in DLC project
dirs = [f for f in os.listdir(os.path.join(dlc_dir, "labeled-data")) if not f.startswith('.') if not f.endswith('_labeled')]


dirs.sort()
dfs = []
for d in dirs:
    print(d)
    try:
        csv_file = glob.glob(os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.csv"))[0]
        df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        if len(df_tmp.index.unique()) != df_tmp.shape[0]:
            # new DLC labeling scheme that splits video/image in different cells
            vids = df_tmp.loc[
                   :, ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2")]
            imgs = df_tmp.loc[
                   :, ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2")]
            new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
            df_tmp1 = df_tmp.drop(
                ("Unnamed: 1_level_0", "Unnamed: 1_level_1", "Unnamed: 1_level_2"), axis=1,
            )
            df_tmp2 = df_tmp1.drop(
                ("Unnamed: 2_level_0", "Unnamed: 2_level_1", "Unnamed: 2_level_2"), axis=1,
            )
            df_tmp2.index = new_col
            df_tmp = df_tmp2
    except IndexError:
        try:
            h5_file = glob.glob(os.path.join(dlc_dir, "labeled-data", d, "CollectedData*.h5"))[0]
            df_tmp = pd.read_hdf(h5_file)
            if type(df_tmp.index) == pd.core.indexes.multi.MultiIndex:
                # new DLC labeling scheme that splits video/image in different cells
                imgs = [i[2] for i in df_tmp.index]
                vids = [df_tmp.index[0][1] for _ in imgs]
                new_col = [f"labeled-data/{v}/{i}" for v, i in zip(vids, imgs)]
                df_tmp1 = df_tmp.reset_index().drop(
                    columns="level_0").drop(columns="level_1").drop(columns="level_2")
                df_tmp1.index = new_col
                df_tmp = df_tmp1
        except IndexError:
            print(f"Could not find labels for {d}; skipping")
    dfs.append(df_tmp)
df_all = pd.concat(dfs)

os.makedirs(lp_dir, exist_ok=True)

# save concatenated labels
df_all.to_csv(os.path.join(lp_dir, "CollectedData.csv"))

# copy frames over
src = os.path.join(dlc_dir, "labeled-data")
dst = os.path.join(lp_dir, "labeled-data")
shutil.copytree(src, dst)

# copy videos over
src = os.path.join(dlc_dir, "videos")
dst = os.path.join(lp_dir, "videos")
if os.path.exists(src):
    print("copying video files")
    shutil.copytree(src, dst)
else:
    print("DLC video directory does not exist; creating empty video directory")
    os.makedirs(dst, exist_ok=True)

# check
for im in df_all.index:
    assert os.path.exists(os.path.join(lp_dir, im))
