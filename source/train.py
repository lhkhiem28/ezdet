
import os, sys
from libs import *
from data import DetImageDataset
from engines import train_fn

datasets = {
    "train":DetImageDataset(
        images_path = "../datasets/VOC2007/train/images", labels_path = "../datasets/VOC2007/train/labels", 
        image_size = 416, 
    ), 
    "val":DetImageDataset(
        images_path = "../datasets/VOC2007/val/images", labels_path = "../datasets/VOC2007/val/labels", 
        image_size = 416, 
    ), 
}
train_loaders = {
    "train":torch.utils.data.DataLoader(
        datasets["train"], collate_fn = datasets["train"].collate_fn, 
        num_workers = 8, batch_size = 32, 
        shuffle = True
    ), 
    "val":torch.utils.data.DataLoader(
        datasets["val"], collate_fn = datasets["val"].collate_fn, 
        num_workers = 8, batch_size = 32, 
        shuffle = True
    ), 
}
model = Darknet("nets/yolov3.cfg")
model.load_darknet_weights("../ckps/darknet53.conv.74")
optimizer = optim.Adam(
    model.parameters(), 
    lr = model.hyperparams["lr"], weight_decay = model.hyperparams["weight_decay"], 
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    eta_min = 0.01*model.hyperparams["lr"], T_max = int(0.9*int(model.hyperparams["num_epochs"])), 
)

wandb.login()
wandb.init(
    # mode = "disabled", 
    project = "ezdet", name = "yolov3", 
)
save_ckp_dir = "../ckps/VOC2007"
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