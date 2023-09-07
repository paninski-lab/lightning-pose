import os, glob, sys, shutil

import scipy as scp
from scipy.io import loadmat
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import trange, tqdm
from copy import deepcopy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""    7M dataset to lightning-pose:
Dataset posted on 2021-02-08, 21:00 authored by Jesse D. Marshall, Diego Aldarondo, William Wang, Bence P. Ölveczky, Timothy Dunn
All videos for Subject 3, Day 1 in Rat 7M.
The video naming scheme is s{subject_id}-d{recording_day}-camera{camera_id}-{starting_frame_idx}.mp4.
Subject ID, recording day, and camera ID match videos to the data in the motion capture and camera calibration parameter .mat files.
Videos are provided in 3500 frame chunks, with the index of the starting frame in each file denoted by the {starting_frame_idx} portion of the filename.
Using the 'frames' field in the 'cameras' struct inside 'mocap.mat', the corresponding video file and frame index can be calculated by used frame_idx // 3500 to get the {starting_frame_idx} and frame_idx % 3500 to get the frame in that file.
"""

path_glob= "/mnt/scratch2/farzad/7m"

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)

def d7M(path = path_glob):
    # list all the labels/other data
    motion_files = glob.glob(os.path.join(path, "motion", "*.mat"))
    video_files = glob.glob(os.path.join(path, "videos", "*.mp4"))
    
    df = pd.DataFrame(columns=["video", "subject_id", "recording_day", 
                               "camera_id", "starting_frame_idx", "label_file"])        
    
    lookup_table_motion_files = [None]*6
    for i in range(6):
        lookup_table_motion_files[i] = [None]*3

    print("Loading Mat files ...")
    for motion_file in tqdm(motion_files):
        
        motion_file_buf = motion_file.split("/")[-1].replace(".mat", "").split("-")
        # print(motion_file,"  !loaded!")
        subject = int(motion_file_buf[1].replace("s", ""))
        day = int(motion_file_buf[2].replace("d", ""))
        lookup_table_motion_files[subject][day] = loadmat(motion_file)
        # print("* ", subject, day, lookup_table_motion_files[subject][day]["name"])
    
    # for subject in range(6):
    #     for day in range(3):
    #         if not lookup_table_motion_files[subject][day] is None:                
    #             print(subject, day, lookup_table_motion_files[subject][day]["name"])
        


    for i, video in tqdm(enumerate(video_files)):
        video_s = video.replace(".mp4", "").split("/")[-1].split("-")
        subject = int(video_s[0].replace("s", ""))
        day = int(video_s[1].replace("d", ""))
        camera = int(video_s[2].replace("camera", ""))
        frame = int(video_s[3])
        # print(subject, day, camera, video_s)
        df.loc[len(df.index)] = [ video,
            subject,
            day,
            camera,
            frame,            
            os.path.join(path, "motion", f"mocap-s{subject}-d{day}.mat")]   
        # print(subject, day, lookup_table_motion[subject][day])  
           
    # df=df.sort_values(["subject_id", "recording_day", "camera_id"])

    print("reading a matfile for bodyparts")
    matfile = loadmat(df["label_file"][0])    
    bodyparts = list(matfile["mocap"].dtype.fields.keys())
    
    print("converting 3D coordinates into 2d space planar camera!")
    # load all motion files so that it would run faster
    positions = trcking_cord_to_lightning_pose(lookup_table_motion_files)
    # print(positions)

    df.to_excel(os.path.join(path, "df2.xlsx"))
    
    # print(camera_number)
    return positions, df, bodyparts

def to_csv(positions, bodyparts, select_camera=[1, 2, 3, 4, 5, 6], select_subject=1, select_day=1, samples=500, path=path_glob):
    bodyparts.sort()
    bodyparts_ = bodyparts * 2
    bodyparts_.sort()
    df_imgs = {}
    rng = np.random.default_rng() # to avoid repitition
    sample_indx = rng.choice(positions[select_subject][1]["Camera1"]["frame"].shape[1], size=samples, replace=False)
    for camera in select_camera:
        df_imgs[camera] = pd.DataFrame(index=range(sample_indx.shape[0]), columns=["scorer", *["7m"]*2*len(bodyparts)])
    print("writing coordinates into the dataframe:")      
    for subject, days in positions.items():
        if len(days)==0:
            continue
        if subject == select_subject:
            for day, cameras in days.items():
                if select_day == day:                    
                    for camera, data in cameras.items():
                        camera = int(camera.replace("Camera", ""))
                        if camera in select_camera:
                            counter = 0
                            data["frame"] = data["frame"].reshape(-1)
                            pbar = tqdm(enumerate(data["frame"][sample_indx]))
                            for _, frame in pbar:
                                # if np.random.rand() < skip_prob:
                                #     continue
                                try:
                                    img_name = f"{subject}_{day}_{camera}_{frame}.png"
                                    # image_path_relative = os.path.join("labeled-data", img_name)
                                    # image_path = os.path.join(path, image_path_relative)
                                    # if not os.path.exists(image_path):
                                    #     raise FileExistsError()
                                    row_buf = [img_name]
                                    frame_indx = np.argwhere(data["frame"] == frame)[0]
                                    for body_part in bodyparts:
                                        row_buf.append(data[body_part][frame_indx, 0][0])
                                        row_buf.append(data[body_part][frame_indx, 1][0])
                                    df_imgs[camera].loc[counter] = deepcopy(row_buf)
                                    counter += 1
                                except FileExistsError:
                                    # print("Does not exist: ", image_path)
                                    continue

    # for camera in select_camera:
    #     df_imgs[camera].to_csv(f"/home/farzad/projects/{camera}.csv")

    # for camera in select_camera:
    #     for indx, frame_name in enumerate(df_imgs[camera]["scorer"]):
    #         frame_path_dst = os.path.join(path, "labeled-data", frame_name)
    #         plt.imshow(mpimg.imread(frame_path_dst))
    #         row_buf = df_imgs[camera].iloc[indx]
    #         for i in range(20):
    #             plt.plot(row_buf[2*i+1], row_buf[2*i+2], 'b*')
    #         plt.savefig("img.png")
    #         plt.close()
    #         input()


    # remove duplicate files 

    print("files exist? ..")
    existance_mtx = np.zeros((len(select_camera), len(df_imgs[select_camera[0]])))
    for camera_index, camera in enumerate(select_camera): 
        # print(camera)
        for frame_indx, image_path_relative in tqdm(enumerate(df_imgs[camera]["scorer"])):
            if not os.path.exists(os.path.join(path, os.path.join("labeled-data", image_path_relative))):
                existance_mtx[camera_index, frame_indx] = 1

    # sample_indx = np.random.randint(low=0, high=len(df_imgs[select_camera[0]]), size=samples)
    print("saving ...")
    indices = np.argwhere(existance_mtx.sum(axis=0)==0).reshape(-1)

    data_folder = "data"
    for camera in select_camera:
        os.makedirs(os.path.join(path, data_folder, "labeled-data", str(camera)), exist_ok=True)    
    for camera in select_camera:
        # df_imgs[camera] = df_imgs[camera][sample_indx]
        # df_buf = deepcopy(df_imgs[camera].iloc[indices])
        for indx, frame_name in enumerate(df_imgs[camera].iloc[indices]["scorer"]):
            frame_path = os.path.join(path, "labeled-data", frame_name)
            frame_path_dst = os.path.join(path, f"{data_folder}/labeled-data/{camera}/{frame_name}")
            shutil.copyfile(os.path.join(path, frame_path), frame_path_dst)

        df_buf = deepcopy(df_imgs[camera].iloc[indices])
        df_buf["scorer"] = f"labeled-data/{camera}/" + df_buf["scorer"]
        df_buf.iloc[0] = ["bodyparts", *bodyparts_]
        df_buf.iloc[1] = ["coords", *["x", "y"]*len(bodyparts)]
        csv_path = os.path.join(path, data_folder, f"{camera}.csv")
        print(csv_path)
        df_buf.replace(to_replace='[nan]', value=np.nan, inplace=True)
        df_buf.to_csv(csv_path, index=False)
    return df_imgs

def trcking_cord_to_lightning_pose(lookup_table_motion_files):
    positions = {}
    # print("trcking_cord_to_lightning_pose")
    for subject in trange(6):
        positions[subject] = {}
        for day in range(3):
            if not lookup_table_motion_files[subject][day] is None:
                positions[subject][day] = {}
                for cam_num, camera in enumerate(list(lookup_table_motion_files[subject][day]["cameras"].dtype.fields.keys())):
                    positions[subject][day][camera] = {}
                    camera_params = lookup_table_motion_files[subject][day]["cameras"][0,0][cam_num]
                    frame = camera_params[0][0][0]
                    IntrinsicMatrix=camera_params[0][0][1]
                    rotationMatrix=camera_params[0][0][2]
                    translationVector=camera_params[0][0][3]
                    TangentialDistortion=camera_params[0][0][4]
                    RadialDistortion=camera_params[0][0][5]
                    positions[subject][day][camera]["frame"] = frame
                    for i, body_part in enumerate(list(lookup_table_motion_files[subject][day]["mocap"].dtype.fields.keys())):
                        xyz = lookup_table_motion_files[subject][day]["mocap"][0,0][i]
                        c3d = np.concatenate((xyz, np.ones((xyz.shape[0],1))), axis=1).T
                        uvw = IntrinsicMatrix.T @ np.concatenate((rotationMatrix.T, translationVector.T), axis=1) @ c3d
                        uvw = uvw/uvw[2, :]
                        uvw = uvw.T
                        # uvw = IntrinsicMatrix * np.concatenate((xyz, np.ones(xyz.shape[0], 1))) * xyz
                        # print(subject, day, camera, body_part, xyz.shape, uvw)
                        # return uvw[:, 0:2]
                        positions[subject][day][camera][body_part] = deepcopy(uvw[:, 0:2])
    return positions


def extract_img_batch(df, output_directory = path_glob):
    output_directory = os.path.join(output_directory,"labeled-data")
    
    # Create a multiprocessing pool with the number of available CPU cores
    num_processes = mp.cpu_count()
    print(num_processes)
    processes = []
    # Process each video in parallel    
    for video_path, subject, day, camera, frame_offset in zip(df["video"].values, df["subject_id"].values, df["recording_day"].values, df["camera_id"].values, df["starting_frame_idx"].values):        
        p = mp.Process(target=extract_img, args=(video_path, subject, day, camera, frame_offset, output_directory))
        # p.start()
        processes.append(p)

    # Wait for all processes to finish
    n=num_processes-4
    print(len(processes)/n)
    for i in trange(0, len(processes), n):
        for j in range(n):
            # print(i+j)
            processes[i+j].start()
        print("batch: ", i)
        for j in range(n):
            processes[i+j].join()
            processes[i+j].close()

def extract_img(input_video_path, subject, day, camera, frame_offset, output_directory):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Initialize variables
    # frame_offset = 0
    # Loop through each frame in the video
    while True:
        ret, frame = cap.read()
        # frame = cv2.convertScaleAbs(frame, alpha=(255.0))
        
        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Save the frame as an image
        output_image_path = os.path.join(output_directory, f"{subject}_{day}_{camera}_{frame_offset}.png")
        cv2.imwrite(output_image_path, frame)
        
        frame_offset += 1

    # Release the video file and close any open windows
    cap.release()
    # cv2.destroyAllWindows()          
    return True      

def vid_gray_batch(df):
    
    # Create a multiprocessing pool with the number of available CPU cores
    num_processes = mp.cpu_count()
    print(num_processes)
    processes = []

    # Process each video in parallel
    output_directory = os.path.join(path_glob, "gray")
    for video_path in df["video"].values:
        p = mp.Process(target=vid_gray, args=(video_path, output_directory, ))
        # p.start()
        processes.append(p)

    # Wait for all processes to finish
    n=num_processes
    print(len(processes)/n)
    for i in trange(0, len(processes), n):
        for j in range(n):
            processes[i+j].start()
        print("batch: ", i)

        for j in range(n):
            processes[i+j].join()

def vid_gray(input_path, output_path, color="gray"):
    cap = cv2.VideoCapture(input_path)

    output_path = os.path.join(output_path, input_path.split("/")[-1])

    if not cap.isOpened():
        print("Error: Unable to open the video.")
        return    
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    # codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        if color == "gray":
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(gray_frame)
        else:
            out.write(frame)
        # Write the grayscale frame to the output video
        

    # Release video capture and writer objects
    cap.release()
    out.release()


if __name__ == "__main__":
    
    path_glob= "/mnt/scratch2/farzad/7m"
    # first get all the tables for data processing
    positions, df, bodyparts = d7M()
    # print(df.head())

    # run this forconvertin rgb videos to gray scale
    # vid_gray_batch(df)
    # vid_gray("/mnt/scratch2/farzad/7m/videos/s1-d1-camera1-56000.mp4", "/mnt/scratch2/farzad/7m/cam", color="rgb")

    # use this to extract all the images from the videos
    extract_img_batch(df, output_directory="/mnt/scratch2/farzad/7m")    

    # run this to sample the images and create the csv files (first extract all the images from the videos)
    df_imgs = to_csv(positions, bodyparts, select_camera=[1,2,3,4,5,6], select_subject=1, samples=200, path=path_glob)

