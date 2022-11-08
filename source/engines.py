
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

    torch.save(model, "{}/best.ptl".format(save_ckp_dir))