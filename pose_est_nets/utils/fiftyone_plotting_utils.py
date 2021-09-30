import fiftyone as fo
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import imgaug.augmenters as iaa
import torch


# def create_gt_dataset(cfg: DictConfig) -> None:
#     csv_data = pd.read_csv(
#         os.path.join(cfg.data.data_dir, cfg.data.csv_path),
#         header=OmegaConf.to_object(cfg.data.header_rows),
#     )
#     image_names = list(csv_data.iloc[:, 0])

#     img_path = os.path.join(cfg.data.data_dir, image_names[1])
#     gt_keypoints = csv_data.iloc[:, 1:].to_numpy()
#     samples = []
#     for idx, img_name in enumerate(image_names):
#         kpts = gt_keypoints[idx]
#         kpts = kpts.reshape(-1, 2)
#         kpts_list = []
#         for i in range(len(kpts)):
#             kpts_list.append(tuple(kpts[i]))
#         img_path = os.path.join(cfg.data.data_dir, img_name)
#         print(img_path)
#         assert os.path.isfile(img_path)
#         sample = fo.Sample(filepath=img_path)
#         sample["ground_truth"] = fo.Keypoints(keypoints=[fo.Keypoint(points=kpts_list)])
#         samples.append(sample)
#         # for testing purposes
#         if idx > 20:
#             break

#     gt_dataset = fo.Dataset("mouse_data")
#     gt_dataset.add_samples(samples)
#     print(gt_dataset)
#     return gt_dataset


def evaluate(
    cfg, datamod, best_model
):  # removed trainer arg, and assuming best_model is given

    # Create a dataset from a directory of images
    path_to_ims = os.path.join(cfg.data.data_dir, "barObstacleScaling1")
    assert os.path.isdir(path_to_ims)
    gt_dataset = fo.Dataset.from_images_dir(path_to_ims)
    test_indices = datamod.test_set.indices
    best_model.run_subpixelmaxima(pred)
    # for idx, sample in enumerate(gt_dataset):
    for idx, sample in enumerate(gt_dataset.iter_samples(progress=True)):
        img_kpts = datamod.fulldataset.labels[idx]  # a list of tensors.
        img_kpts_list = []
        for i in range(len(img_kpts)):  # iterating over the list
            if ~torch.isnan(
                img_kpts[i]
            ).all():  # only if there are no nans in the length-2 tensor, we add keypoints for visualization
                img_kpts_list.append(
                    tuple((float(img_kpts[i][0] / 406), float(img_kpts[i][1] / 396)))
                )
        # print(img_kpts_list)
        sample["ground_truth"] = fo.Keypoints(
            keypoints=[fo.Keypoint(label="square", points=img_kpts_list)]
        )
        sample.save()
    # new_samples.append(sample)
    # print(sample)
    # new_dataset = fo.Dataset("labeled_data")
    # new_dataset.add_samples(new_samples)
    # print(sample)

    # print(gt_dataset)
    # print(new_dataset)
    # gt_dataset = create_gt_dataset(cfg)
    # print(gt_dataset.persistent = True)
    print("counts: {} ".format(gt_dataset.count("ground_truth.keypoints.points")))
    print("metadata: {} ".format(gt_dataset.compute_metadata()))
    print("first: {} ".format(gt_dataset.first()))

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
    best_model.eval()
    for idx, sample in enumerate(gt_dataset):
        img_kpts = datamod.fulldataset.labels[idx]
        img_kpts_list = []
        for i in range(len(img_kpts)):
            #img_kpts_list.append(tuple((float(img_kpts[i][0]),float(img_kpts[i][1]))))
            img_kpts_list.append(tuple((float(img_kpts[i][0]/406),float(img_kpts[i][1]/396))))
        sample["ground_truth"] = fo.Keypoints(keypoints=[fo.Keypoint(label="square", points=img_kpts_list)])
        if idx in test_indices:
            img = datamod.fulldataset.__getitem__(idx)[0].unsqueeze(0)
            pred = best_model.forward(img)
            if cfg.data.data_type == "heatmap":
                pred = best_model.run_subpixelmaxima(pred)
            #print(pred)
            #print(pred.shape)
            print(img.numpy().shape)
            resized_pred = reverse_transform(
                images=img.numpy()[0,0], #WHY?
                keypoints=pred.numpy()[np.newaxis,...].reshape((1, -1, 2)),
            )
            print(resized_pred[0].shape)
            resized_pred = resized_pred[1]
            print(resized_pred)
            resized_pred = resized_pred[0]  # get rid of the batch dimension
            kpts_list = []
            print(resized_pred.shape)
            for i in range(len(resized_pred)):
                kpts_list.append(tuple((float(resized_pred[i][0]/406),float(resized_pred[i][1]/396))))
            sample["predictions"] = fo.Keypoints(keypoints=[fo.Keypoint(points=kpts_list)])
            print(sample)
        sample.save()
    results = gt_dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="predictions",
    )
    #print(results)
    results.print_report
    gt_dataset.persistent = True
    session = fo.launch_app(gt_dataset, remote=True)
    # session.view = gt_dataset.exclude_fields("ground_truth").take(10) # suggested by forum
    # random_view = gt_dataset.take(10)
    # session.view = random_view
    session.wait()
    # session = fo.launch_app(gt_dataset, remote=True)


def plot():  # ?
    return
