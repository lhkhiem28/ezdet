
import os, sys
from libs import *
from data import DetImageDataset
from engines import train_fn

datasets = {
    "train":DetImageDataset(
        images_path = "../datasets/VOC2007/train/images", labels_path = "../datasets/VOC2007/train/labels", 
    ), 
    "val":DetImageDataset(
        images_path = "../datasets/VOC2007/train/images", labels_path = "../datasets/VOC2007/train/labels", 
    ), 
}
train_loaders = {
    "train":torch.utils.data.DataLoader(
        datasets["train"], collate_fn = datasets["train"].collate_fn, 
        num_workers = 8, batch_size = 48, 
        shuffle = True
    ), 
    "val":torch.utils.data.DataLoader(
        datasets["val"], collate_fn = datasets["val"].collate_fn, 
        num_workers = 8, batch_size = 48, 
        shuffle = True
    ), 
}
model = Darknet("nets/yolov3.cfg")
model.load_darknet_weights("nets/darknet53.conv.74")
optimizer = optim.Adam(
    model.parameters(), 
    lr = model.hyperparams["lr"], weight_decay = model.hyperparams["weight_decay"], 
)

save_ckp_dir = "../ckps/VOC2007"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, 
    model, 
    num_epochs = 300, 
    optimizer = optimizer, 
    save_ckp_dir = save_ckp_dir, 
)