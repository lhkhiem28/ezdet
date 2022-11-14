
import os, sys
from libs import *
from data import DetImageDataset
from engines import train_fn

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
args = parser.parse_args()

datasets = {
    "train":DetImageDataset(
        images_path = "../datasets/{}/train/images".format(args.dataset), labels_path = "../datasets/{}/train/labels".format(args.dataset)
        , image_size = 416
        , augment = True
        , multiscale = True
    ), 
    "val":DetImageDataset(
        images_path = "../datasets/{}/val/images".format(args.dataset), labels_path = "../datasets/{}/val/labels".format(args.dataset)
        , image_size = 416
        , augment = False
        , multiscale = False
    ), 
}
train_loaders = {
    "train":torch.utils.data.DataLoader(
        datasets["train"], collate_fn = datasets["train"].collate_fn, 
        num_workers = 8, batch_size = 32, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        datasets["val"], collate_fn = datasets["val"].collate_fn, 
        num_workers = 8, batch_size = 32, 
        shuffle = False, 
    ), 
}
model = Darknet("pytorchyolo/configs/yolov3.cfg")
model.load_darknet_weights("../ckps/darknet53.conv.74")
optimizer = optim.Adam(
    model.parameters(), 
    lr = model.hyperparams["lr"], weight_decay = model.hyperparams["weight_decay"], 
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    eta_min = 0.01*model.hyperparams["lr"], T_max = int(0.92*int(model.hyperparams["num_epochs"])), 
)

wandb.login()
wandb.init(
    project = "ezdet", name = args.dataset, 
)
save_ckp_dir = "../ckps/{}".format(args.dataset)
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, 
    model, 
    num_epochs = int(model.hyperparams["num_epochs"]), 
    optimizer = optimizer, 
    lr_scheduler = lr_scheduler, 
    save_ckp_dir = save_ckp_dir, 
)
wandb.finish()