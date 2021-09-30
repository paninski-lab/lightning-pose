import fiftyone as fo
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from pose_est_nets.models.new_heatmap_tracker import HeatmapTracker
import torch


#PLAN: reads in hydra cfg, datamodule, and best_model (could be changed to list of best models)
#Goes through the whole dataset using datamodule.fulldataset
#   For each image in the dataset, create a sample with the image path, annotate images corresponding to train/valid/test,
#   Add ground truth predictions from fulldataset.labels (filter out the nans)
#   For each model (one for now), make predictions on the data, check if subpixelmaxima has to be taken (if dataset is heatmap), 
#   and resize predictions back to original image dimension,
#   and get rid of predictions of keypoints corresponding to nans in the ground truth
#   add sample to sample list 

def tensor_to_keypoint_list( #TODO: move to utils file
    keypoint_tensor
):
    img_kpts_list = []
    for i in range(len(keypoint_tensor)):
        img_kpts_list.append(tuple((float(keypoint_tensor[i][0]/406),float(keypoint_tensor[i][1]/396))))
        #keypoints are normalized to the original image dims, either add these to data config, or automatically detect by
        #loading a sample image in dataset.py or something
    return img_kpts_list


def make_dataset_and_evaluate(
    cfg, datamod, best_models
):
    reverse_transform = []
    reverse_transform.append(
        iaa.Resize(
            {
                "height": 406,  # HARDCODED FOR NOW,
                "width": 396,
            }
        )
    )
    reverse_transform = iaa.Sequential(reverse_transform)
    image_names = datamod.fulldataset.image_names
    gt_keypoints = datamod.fulldataset.labels
    samples = []
    train_indices = datamod.train_set.indices
    valid_indices = datamod.valid_set.indices
    test_indices = datamod.test_set.indices
    for model in best_model.values():
        model.eval()
    for idx, img_name in enumerate(image_names):
        print(idx)
        img_path = os.path.join(cfg.data.data_dir, img_name)
        #assert os.path.isfile(img_path)
        gt_img_kpts = gt_keypoints[idx]
        nan_bool = (
            torch.sum(torch.isnan(gt_img_kpts), dim=1) > 0
        )  # e.g., when dim == 0, those columns (keypoints) that have more than zero nans
        gt_img_kpts = gt_img_kpts[~nan_bool]
        gt_kpts_list = tensor_to_keypoint_list(gt_img_kpts)
        if idx in train_indices:
            tag = "train"
        elif idx in valid_indices:
            tag = "valid"
        elif idx in test_indices:
            tag = "test"
        else:
            tag = "error"
        sample = fo.Sample(filepath=img_path, tags=[tag])
        sample["ground_truth"] = fo.Keypoints(keypoints=[fo.Keypoint(points=gt_kpts_list)])
        img = datamod.fulldataset.__getitem__(idx)[0].unsqueeze(0)
        img_BHWC = img.permute(0, 2, 3, 1) #Needs to be BHWC format       
        for name, model in best_models.items():
            pred = model.forward(img)
            if isinstance(model, HeatmapTracker) or issubclass(type(model), HeatmapTracker): #check if model is in the heatmap family
                pred = model.run_subpixelmaxima(pred)
            resized_pred = reverse_transform(
                images=img_BHWC.numpy(),
                keypoints=(pred.detach().numpy().reshape((1, -1, 2)))
            )[1][0]
            pred_kpts_list = tensor_to_keypoint_list(resized_pred[~nan_bool])
            sample[name+"_prediction"] = fo.Keypoints(keypoints=[fo.Keypoint(points=pred_kpts_list)])
        samples.append(sample)
        #print(sample)
        
    full_dataset = fo.Dataset("mouse_data")
    full_dataset.add_samples(samples)
    print(full_dataset)
    print("counts: {} ".format(full_dataset.count("ground_truth.keypoints.points")))
    print("metadata: {} ".format(full_dataset.compute_metadata()))
    print("first: {} ".format(full_dataset.first()))
    session = fo.launch_app(full_dataset, remote=True)
    session.wait()
    return



# def evaluate(
#     cfg, datamod, best_model
# ):  # removed trainer arg, and assuming best_model is given

#     # Create a dataset from a directory of images
#     path_to_ims = os.path.join(cfg.data.data_dir, "barObstacleScaling1")
#     assert os.path.isdir(path_to_ims)
#     gt_dataset = fo.Dataset.from_images_dir(path_to_ims)
#     test_indices = datamod.test_set.indices
#     best_model.run_subpixelmaxima(pred) 
#     # for idx, sample in enumerate(gt_dataset):
#     for idx, sample in enumerate(gt_dataset.iter_samples(progress=True)):
#         img_kpts = datamod.fulldataset.labels[idx]  # a list of tensors.
#         img_kpts_list = []
#         for i in range(len(img_kpts)):  # iterating over the list
#             if ~torch.isnan(
#                 img_kpts[i]
#             ).all():  # only if there are no nans in the length-2 tensor, we add keypoints for visualization
#                 img_kpts_list.append(
#                     tuple((float(img_kpts[i][0] / 406), float(img_kpts[i][1] / 396)))
#                 )
#         # print(img_kpts_list)
#         sample["ground_truth"] = fo.Keypoints(
#             keypoints=[fo.Keypoint(label="square", points=img_kpts_list)]
#         )
#         sample.save()
#     # new_samples.append(sample)
#     # print(sample)
#     # new_dataset = fo.Dataset("labeled_data")
#     # new_dataset.add_samples(new_samples)
#     # print(sample)

#     # print(gt_dataset)
#     # print(new_dataset)
#     # gt_dataset = create_gt_dataset(cfg)
#     # print(gt_dataset.persistent = True)
#     print("counts: {} ".format(gt_dataset.count("ground_truth.keypoints.points")))
#     print("metadata: {} ".format(gt_dataset.compute_metadata()))
#     print("first: {} ".format(gt_dataset.first()))

#     reverse_transform = []
#     reverse_transform.append(
#         iaa.Resize(
#             {
#                 "height": 406,  # HARDCODED FOR NOW,
#                 "width": 396,
#             }
#         )
#     )
#     reverse_transform = iaa.Sequential(reverse_transform)
#     best_model.eval()
#     for idx, sample in enumerate(gt_dataset):
#         img_kpts = datamod.fulldataset.labels[idx]
#         img_kpts_list = []
#         for i in range(len(img_kpts)):
#             #img_kpts_list.append(tuple((float(img_kpts[i][0]),float(img_kpts[i][1]))))
#             img_kpts_list.append(tuple((float(img_kpts[i][0]/406),float(img_kpts[i][1]/396))))
#         sample["ground_truth"] = fo.Keypoints(keypoints=[fo.Keypoint(label="square", points=img_kpts_list)])
#         if idx in test_indices:
#             img = datamod.fulldataset.__getitem__(idx)[0].unsqueeze(0)
#             pred = best_model.forward(img)
#             if cfg.data.data_type == "heatmap":
#                 pred = best_model.run_subpixelmaxima(pred)
#             #print(pred)
#             #print(pred.shape)
#             print(img.numpy().shape)
#             resized_pred = reverse_transform(
#                 images=img.numpy()[0,0], #WHY?
#                 keypoints=pred.numpy()[np.newaxis,...].reshape((1, -1, 2)),
#             )
#             print(resized_pred[0].shape)
#             resized_pred = resized_pred[1]
#             print(resized_pred)
#             resized_pred = resized_pred[0]  # get rid of the batch dimension
#             kpts_list = []
#             print(resized_pred.shape)
#             for i in range(len(resized_pred)):
#                 kpts_list.append(tuple((float(resized_pred[i][0]/406),float(resized_pred[i][1]/396))))
#             sample["predictions"] = fo.Keypoints(keypoints=[fo.Keypoint(points=kpts_list)])
#             print(sample)
#         sample.save()
#     results = gt_dataset.evaluate_detections(
#         "predictions",
#         gt_field="ground_truth",
#         eval_key="predictions",
#     )
#     #print(results)
#     results.print_report
#     gt_dataset.persistent = True
#     session = fo.launch_app(gt_dataset, remote=True)
#     # session.view = gt_dataset.exclude_fields("ground_truth").take(10) # suggested by forum
#     # random_view = gt_dataset.take(10)
#     # session.view = random_view
#     session.wait()
#     # session = fo.launch_app(gt_dataset, remote=True)
