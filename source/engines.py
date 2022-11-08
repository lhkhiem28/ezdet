
import os, sys
from libs import *

def train_fn(
    train_loaders, 
    model, 
    num_epochs, 
    optimizer, 
    save_ckp_dir = "./", 
    training_verbose = True, 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()

    for epoch in tqdm.tqdm(range(1, num_epochs + 1), disable = training_verbose):
        if training_verbose:print("epoch {:2}/{:2}".format(epoch, num_epochs) + "\n" + "-"*16)
        model.train()
        running_loss = 0.0
        for images, labels in tqdm.tqdm(train_loaders["train"], disable = not training_verbose):
            images, labels = images.cuda(), labels.cuda()

            logits = model(images)
            loss = compute_loss(
                logits, labels, 
                model, 
            )[0]

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss = running_loss + loss.item()*images.size(0)
        epoch_loss = running_loss/len(train_loaders["train"].dataset)
        if training_verbose:
            print("{:<5} - loss:{:.4f}".format(
                "train", 
                epoch_loss, 
            ))

        with torch.no_grad():
            model.eval()
            running_classes, running_statistics = [], []
            for images, labels in tqdm.tqdm(train_loaders["val"], disable = not training_verbose):
                images, labels = images.cuda(), labels.cuda()
                labels[:, 2:] = xywh2xyxy(labels[:, 2:])
                labels[:, 2:] = labels[:, 2:]*int(train_loaders["val"].dataset.image_size[0])

                logits = model(images)
                logits = non_max_suppression(
                    logits, 
                    conf_thres = 0.1, iou_thres = 0.5, 
                )

                running_classes, running_statistics = running_classes + labels[:, 1].tolist(), running_statistics + get_batch_statistics(
                    [elem.cpu() for elem in logits], labels.cpu(), 
                    0.5, 
                )
        epoch_map = ap_per_class(
           *[
                np.concatenate(stats, 0) for stats in list(zip(*running_statistics))
            ], running_classes
        )[2].mean()
        if training_verbose:
            print("{:<5} -  map:{:.4f}".format(
                "val", 
                epoch_map, 
            ))

    torch.save(model, "{}/best.ptl".format(save_ckp_dir))