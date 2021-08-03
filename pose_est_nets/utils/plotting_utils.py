import imgaug.augmenters as iaa
import numpy as np
import torch
from pose_est_nets.utils.IO import set_or_open_folder, get_latest_version
import matplotlib.pyplot as plt
import os


def saveNumericalPredictions(model, datamod, threshold):
    i = 0
    # hardcoded for mouse data
    rev_augmenter = []
    rev_augmenter.append(
        iaa.Resize({"height": 406, "width": 396})
    )  # get rid of this for the fish
    rev_augmenter = iaa.Sequential(rev_augmenter)
    model.eval()
    full_dl = datamod.full_dataloader()
    test_dl = datamod.test_dataloader()
    final_gt_keypoints = np.empty(shape=(len(test_dl), model.num_keypoints, 2))
    final_imgs = np.empty(shape=(len(test_dl), 406, 396, 1))
    final_preds = np.empty(shape=(len(test_dl), model.num_keypoints, 2))

    # dpk_final_preds = np.empty(shape = (len(test_dl), model.num_keypoints, 2))

    for idx, batch in enumerate(test_dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        if torch.cuda.is_available():
            heatmap_pred = heatmap_pred.cuda()
            y = y.cuda()
        pred_keypoints, y_keypoints = model.computeSubPixMax(heatmap_pred, y, threshold)
        # dpk_final_preds[i] = pred_keypoints
        pred_keypoints = pred_keypoints.cpu()
        y_keypoints = y_keypoints.cpu()
        x = x[:, 0, :, :]  # only taking one image dimension
        x = np.expand_dims(x, axis=3)
        final_imgs[i], final_gt_keypoints[i] = rev_augmenter(
            images=x, keypoints=np.expand_dims(y_keypoints, axis=0)
        )
        final_imgs[i], final_preds[i] = rev_augmenter(
            images=x, keypoints=np.expand_dims(pred_keypoints, axis=0)
        )
        # final_gt_keypoints[i] = y_keypoints
        # final_preds[i] = pred_keypoints
        i += 1

    final_gt_keypoints = np.reshape(
        final_gt_keypoints, newshape=(len(test_dl), model.num_targets)
    )
    final_preds = np.reshape(final_preds, newshape=(len(test_dl), model.num_targets))
    # dpk_final_preds = np.reshape(dpk_final_preds, newshape = (len(test_dl), model.num_targets))

    # np.savetxt('../../preds/mouse_gt.csv', final_gt_keypoints, delimiter = ',', newline = '\n')
    folder_name = get_latest_version("lightning_logs")
    csv_folder = set_or_open_folder(os.path.join("preds", folder_name))

    np.savetxt(
        os.path.join(csv_folder, "preds.csv"), final_preds, delimiter=",", newline="\n"
    )
    # np.savetxt('../preds/dpk_fish_predictions.csv', dpk_final_preds, delimiter = ',', newline = '\n')
    return


def plotPredictions(model, datamod, save_heatmaps, threshold, mode):
    folder_name = get_latest_version("lightning_logs")
    img_folder = set_or_open_folder(os.path.join("preds", folder_name, "images"))

    if save_heatmaps:
        heatmap_folder = set_or_open_folder(
            os.path.join("preds", folder_name, "heatmaps")
        )

    model.eval()
    if mode == "train":
        dl = datamod.train_dataloader()
    else:
        dl = datamod.test_dataloader()
    i = 0
    for idx, batch in enumerate(dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        if save_heatmaps:
            plt.imshow(heatmap_pred[0, 4].detach().cpu().numpy())
            plt.savefig(os.path.join(heatmap_folder, "pred_map_%i" % i + ".png"))
            plt.clf()
            plt.imshow(y[0, 4].detach().cpu().numpy())
            plt.savefig(os.path.join(heatmap_folder, "gt_map_%i" % i + ".png"))
            plt.clf()
        # if torch.cuda.is_available():
        #     heatmap_pred = heatmap_pred.cuda()
        #     y = y.cuda()
        # TODO: that works, but remove cuda calls! threshold is on model which is on cuda, heatmap_pred and y are on CPU after saving heatmaps
        pred_keypoints, y_keypoints = model.computeSubPixMax(
            heatmap_pred.cuda(), y.cuda(), threshold
        )
        plt.imshow(x[0][0])
        pred_keypoints = pred_keypoints.cpu()
        y_keypoints = y_keypoints.cpu()
        plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c="blue")
        plt.scatter(y_keypoints[:, 0], y_keypoints[:, 1], c="orange")
        plt.savefig(os.path.join(img_folder, "pred_%i" % i + ".png"))
        plt.clf()
        i += 1
