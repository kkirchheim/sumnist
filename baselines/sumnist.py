from torch.utils.data import Dataset
from os.path import join
from os import listdir 
from PIL import Image
import numpy as np 
import torch        
import numpy as np 
import torch 
        
class SuMNIST(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.transforms = transforms
        
        if train:
            with np.load(join(root, "x-train.npz")) as data:
                self.x = torch.tensor(data['arr_0'])
                
            with np.load(join(root, "y-train.npz")) as data:
                self.y = torch.tensor(data['arr_0'])
                
            with np.load(join(root, "b-train.npz")) as data:
                self.b = torch.tensor(data['arr_0'])
        else:
            with np.load(join(root, "x-test.npz")) as data:
                self.x = torch.tensor(data['arr_0'])

            with np.load(join(root, "y-test.npz")) as data:
                self.y = torch.tensor(data['arr_0'])
                self.y2 = torch.where(self.y.sum(dim=1) == 20, 0, -1)
                
            with np.load(join(root, "b-test.npz")) as data:
                self.b = torch.tensor(data['arr_0'])
                
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        img = self.x[index]
        img= img.repeat(3, 1, 1) # To RGB 
        
        bboxes = self.b[index]
        
        boxes = []
        for box in bboxes:
            x_min, x_max, y_min, y_max = box 
            boxes.append((x_min, y_min, x_max, y_max))
           
        bboxes = boxes
        # bboxes = [
        #     [ 0,  0, 28, 28],
        #     [ 0, 28, 28, 56],
        #     [ 28, 0, 56, 28],
        #     [28, 28, 56, 56],
        # ]

        labels = self.y[index]

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels =  torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = labels = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
    
        return img, target
