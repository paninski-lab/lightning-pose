import torch
import matplotlib.pyplot as plt
import os
from pose_est_nets.utils.IO import save_object

def get_latest_version(lightning_logs_path):
    # TODO: add what's needed to pull out ckpts
    ints = [int(l.split('_')[-1]) for l in os.listdir(lightning_logs_path)]
    latest_version = lightning_logs_path[ints.index(max(ints))]
    print("version used: %s" %latest_version)
    return latest_version


def predict_plot_test_epoch(model,
                  dataloader,
                  preds_folder):
    # Nick's version commented out
    # model = model_class.load_from_checkpoint(checkpoint_path=args.ckpt, num_targets=34, resnet_version=50, transfer=False)
    #checkpoint_name = checkpoint_path.split('/')[-1].split('.')[0]
    #preds_folder = os.path.join('preds', checkpoint_name)
    preds_dict = {}
    preds_dict["labels"] = []
    preds_dict["preds"] = []

    #model = model_instance.load_from_checkpoint(checkpoint_path)
    model.eval()
    counter = 0
    for batch in dataloader:
        images, labels = batch
        predictions = model(images)

        for i in range(images.shape[0]):
            scatter_predictions(images[i], labels[i], predictions[i])
            plt.savefig(os.path.join(preds_folder, "test_im_" + str(counter) + ".png"))
            plt.close()
            preds_dict["preds"].append(predictions[i])
            preds_dict["labels"].append(labels[i])
            counter+=1

    save_object(preds_dict, os.path.join(preds_folder, "preds"))

    return preds_dict

def scatter_predictions(image: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor) -> None:
    '''we assume that labels/preds come in format [x_1, y_1, x_2, y_2 ...]'''
    plt.figure()
    plt.imshow(image[0])
    plt.scatter(labels[0::2], labels[1::2], label='labels')
    plt.scatter(preds[0::2], preds[1::2], label='preds')

