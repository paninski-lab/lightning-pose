import imgaug.augmenters as iaa
import numpy as np
from pose_est_nets.utils.IO import set_or_open_folder
import matplotlib.pyplot as plt


def saveNumericalPredictions(model, datamod, threshold):
    i = 0
    #hardcoded for mouse data
    rev_augmenter = []
    rev_augmenter.append(iaa.Resize({"height": 406, "width": 396})) #get rid of this for the fish
    rev_augmenter = iaa.Sequential(rev_augmenter)
    model.eval()
    full_dl = datamod.full_dataloader()
    test_dl = datamod.test_dataloader()
    final_gt_keypoints = np.empty(shape = (len(test_dl), model.num_keypoints, 2))
    final_imgs = np.empty(shape = (len(test_dl), 406, 396, 1))
    final_preds = np.empty(shape = (len(test_dl), model.num_keypoints, 2))

    #dpk_final_preds = np.empty(shape = (len(test_dl), model.num_keypoints, 2))

    for idx, batch in enumerate(test_dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        output_shape = data.output_shape #changed to small
        #dpk_pred_keypoints, dpk_y_keypoints = computeSubPixMax(heatmap_pred, y, output_shape, threshold)
        pred_keypoints, y_keypoints = model.computeSubPixMax(heatmap_pred.cuda(), y.cuda(), threshold)
        #dpk_final_preds[i] = pred_keypoints
        pred_keypoints = pred_keypoints.cpu()
        y_keypoints = y_keypoints.cpu()
        x = x[:,0,:,:] #only taking one image dimension
        x = np.expand_dims(x, axis = 3)
        final_imgs[i], final_gt_keypoints[i] = rev_augmenter(images = x, keypoints = np.expand_dims(y_keypoints, axis = 0))
        final_imgs[i], final_preds[i] = rev_augmenter(images = x, keypoints = np.expand_dims(pred_keypoints, axis = 0))
        #final_gt_keypoints[i] = y_keypoints
        #final_preds[i] = pred_keypoints
        i += 1

    final_gt_keypoints = np.reshape(final_gt_keypoints, newshape = (len(test_dl), model.num_targets))
    final_preds = np.reshape(final_preds, newshape = (len(test_dl), model.num_targets))
    #dpk_final_preds = np.reshape(dpk_final_preds, newshape = (len(test_dl), model.num_targets))

    #np.savetxt('../preds/mouse_gt.csv', final_gt_keypoints, delimiter = ',', newline = '\n')
    np.savetxt('../preds/mouse_pca2view_larger_preds.csv', final_preds, delimiter = ',', newline = '\n')
    #np.savetxt('../preds/dpk_fish_predictions.csv', dpk_final_preds, delimiter = ',', newline = '\n')
    return

def plotPredictions(model, datamod, save_heatmaps, threshold, mode):
    if (save_heatmaps):
        heatmap_folder = set_or_open_folder("preds/heatmaps")
    img_folder = set_or_open_folder("preds/images")

    model.eval()
    if mode == 'train':
        dl = datamod.train_dataloader()
    else:
        dl = datamod.test_dataloader()
    i = 0
    for idx, batch in enumerate(dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        if (save_heatmaps):
            plt.imshow(heatmap_pred[0, 4].detach())
            plt.savefig(os.path.join(heatmap_folder, 'pred_map_%i' % i + '.png'))
            plt.clf()
            plt.imshow(y[0, 4].detach())
            plt.savefig(os.path.join(heatmap_folder, 'gt_map_%i' % i + '.png'))
            plt.clf()
        output_shape = data.output_shape #changed from train_data
        #print(heatmap_pred.device, y.device, model.device)
        #exit()
        pred_keypoints, y_keypoints = model.computeSubPixMax(heatmap_pred.cuda(), y.cuda(), threshold)
        plt.imshow(x[0][0])
        pred_keypoints = pred_keypoints.cpu()
        y_keypoints = y_keypoints.cpu()
        plt.scatter(pred_keypoints[:,0], pred_keypoints[:,1], c = 'blue')
        plt.scatter(y_keypoints[:,0], y_keypoints[:,1], c = 'orange')
        plt.savefig(os.path.join(img_folder, "pred_%i" % i + '.png'))
        plt.clf()
        i += 1