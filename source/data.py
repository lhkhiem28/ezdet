
import os, sys
from libs import *

class DetImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        images_path, labels_path, 
        image_size = (320, 320), 
    ):
        self.image_files, self.label_files,  = sorted(glob.glob(images_path + "/*")), sorted(glob.glob(labels_path + "/*")), 
        self.image_size = image_size

    def __len__(self, 
    ):
        return len(self.image_files)

    def square_pad(self, 
        image, 
    ):
        _, h, w,  = image.shape
        gap_pad = np.abs(h - w)
        if h - w > 0:
            pad = (gap_pad // 2, gap_pad - gap_pad // 2, 0, 0)
        else:
            pad = (0, 0, gap_pad // 2, gap_pad - gap_pad // 2)

        image = F.pad(image, pad, value = 0)
        return image, [0] + list(pad)

    def __getitem__(self, 
        index, 
    ):
        image_file, label_file,  = self.image_files[index], self.label_files[index], 
        image = cv2.cvtColor(cv2.imread(image_file), code = cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1)
        _, h, w,  = image.shape
        image, pad = self.square_pad(image)
        _, padded_h, padded_w,  = image.shape

        bboxes = np.loadtxt(label_file).reshape(-1, 5)
        c1, c2, c3, c4,  = w*(bboxes[:, 1] - bboxes[:, 3]/2) + pad[1], w*(bboxes[:, 1] + bboxes[:, 3]/2) + pad[2], h*(bboxes[:, 2] - bboxes[:, 4]/2) + pad[3], h*(bboxes[:, 2] + bboxes[:, 4]/2) + pad[4], 
        bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4],  = ((c1 + c2)/2)/padded_w, ((c3 + c4)/2)/padded_h, bboxes[:, 3]*(w/padded_w), bboxes[:, 4]*(h/padded_h), 
        return image, F.pad(torch.tensor(bboxes), (1, 0, 0, 0), value = 0)

    def collate_fn(self, 
        batch, 
    ):
        images, labels = list(zip(*batch))
        images = torch.stack([F.interpolate(image.unsqueeze(0), self.image_size, mode = "nearest").squeeze(0) for image in images])/255

        labels = [bboxes for bboxes in labels if bboxes is not None]
        for i, bboxes in enumerate(labels):
            bboxes[:, 0] = i
        if len(labels) != 0:
            labels = torch.cat(labels, 0)
        return images, labels