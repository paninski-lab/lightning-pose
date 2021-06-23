import os
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pose_est_nets.models.regression_tracker import RegressionTracker
from pose_est_nets.datasets.datasets import TrackingDataset, TrackingDataModule
from pose_est_nets.callbacks.freeze_unfreeze_callback import FeatureExtractorFreezeUnfreeze
import matplotlib.pyplot as plt
import json
import argparse
import torch

def set_or_open_folder(folder_path: str) -> str:
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("Opened a new folder at: {}".format(folder_path))
    else:
        print("The folder already exists at: {}".format(folder_path))
    return folder_path

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default=None, help="set true to load model from checkpoint")
parser.add_argument("--predict", action='store_true', help="whether or not to generate predictions on test data")
# parser.add_argument("--ckpt", type=str, default="lightning_logs2/version_1/checkpoints/epoch=271-step=12511.ckpt",
#                     help="path to model checkpoint if you want to load model from checkpoint")
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--validation_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--num_gpus", type=int, default=0)
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--early_stop_patience", type=int, default=6)
parser.add_argument("--unfreezing_epoch", type=int, default=50)

args = parser.parse_args()

model = RegressionTracker(num_targets=34, resnet_version=50, transfer=True)

# specific to the mouseRunningData
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1636, 0.1636, 0.1636], std=[0.1240, 0.1240, 0.1240])
])

dataset = TrackingDataset(root_directory=args.data_dir, csv_path='CollectedData_.csv', header_rows=[1, 2],
                          transform=data_transform)

data_module = TrackingDataModule(dataset,
                                 train_batch_size=args.train_batch_size,
                                 validation_batch_size=args.validation_batch_size,
                                 test_batch_size=args.test_batch_size,
                                 num_workers=args.num_workers)

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=args.early_stop_patience, mode="min"
)

transfer_unfreeze_callback = FeatureExtractorFreezeUnfreeze(args.unfreezing_epoch)

callback_list = []
if args.early_stop_patience<100:
    callback_list.append(early_stopping)
if args.unfreezing_epoch>0:
    callback_list.append(transfer_unfreeze_callback)

trainer = pl.Trainer(gpus=args.num_gpus,
                     log_every_n_steps=15,
                     callbacks=callback_list,
                     auto_scale_batch_size=False,
                     check_val_every_n_epoch=10,
                     max_epochs=args.max_epochs)  # auto_scale_batch_size not working

trainer.fit(model=model, datamodule=data_module)

if (args.predict):
    print("Starting to predict test images")
    predictions_folder = set_or_open_folder('preds')
    model.eval()
    trainer.test(model = model, datamodule = data_module)
    model.eval()
    preds = {}
    i = 1
    f = open(os.path.join(predictions_folder, 'predictions.txt'), 'w')
    predict_dl = data_module.test_dataloader()
    for batch in predict_dl:
        if i > 10:
            break
        x, y = batch
        plt.clf()
        out = model.forward(x.to("cuda" if torch.cuda.is_available() else "cpu"))
        plt.imshow(x[0, 0])
        preds[i] = out.numpy().tolist()
        assert(out == out).squeeze().all()
        plt.scatter(out.numpy()[:,0::2], out.numpy()[:,1::2], c = 'blue')
        plt.scatter(y.numpy()[:,0::2], y.numpy()[:,1::2], c = 'orange')
        plt.savefig(os.path.join(predictions_folder, "test" + str(i) + ".png"))
        i += 1
    f.write(json.dumps(preds))
    f.close()
