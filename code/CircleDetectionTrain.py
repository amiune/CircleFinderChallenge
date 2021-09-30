#!/usr/bin/env python
# coding: utf-8

# # Training

# ## Clean Dataset

# The cleaning of the dataset consist of:
# 1. Removing images that are all black
# 2. Removing images that I considerer to big to process with my computer (with or height greater than 3000 pixels)

# In[3]:


import sys
from tqdm import tqdm
import time
import os
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT_DATA_DIR = "../full/train/"

ROOT_DATA_DIR = sys.argv[1]

print(ROOT_DATA_DIR)

def get_directories(path):
    directories = []
    for x in os.walk(path):
        img_id = x[0][len(path):]
        if img_id != "":
            img_id = x[0][len(path):]
            img_path = "".join([ROOT_DATA_DIR, img_id, "/", img_id, "_PAN.tif"])
            with Image.open(img_path).convert("L") as img:
                if img.size[0] < 3000 and img.size[1] < 3000:
                    img = np.array(img)
                    if img.any(axis=-1).sum() > 0: #non black
                        directories.append(img_id)
    return list(sorted(directories))

start_time = time.time()

directories = get_directories(ROOT_DATA_DIR)
total_files = len(directories)
print(len(directories))
print(directories[0:5])

elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# ## Create Pytorch Dataset Class

# In[4]:


import os
import numpy as np
import torch
from PIL import Image
import fiona
import rasterio
import rasterio.mask

class CircleFinderDataset(torch.utils.data.Dataset):
    def __init__(self, images_ids, transforms=None):
        self.transforms = transforms
        self.imgs = images_ids

    def __getitem__(self, idx):
        
        shapes = None
        with fiona.open(ROOT_DATA_DIR + self.imgs[idx] + "/" + self.imgs[idx] + "_anno.geojson", "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        
        with rasterio.open(ROOT_DATA_DIR + self.imgs[idx] + "/" + self.imgs[idx] + "_PAN.tif") as src:
            transform = src.transform
            rev = ~transform
        
        
        
        # load images ad masks
        img_path = "".join([ROOT_DATA_DIR, self.imgs[idx], "/", self.imgs[idx], "_PAN.tif"])
        img = Image.open(img_path).convert("L")

        # get bounding box coordinates for each mask
        num_objs = len(shapes)
        boxes = []
        areas = []
        for shape in shapes:
            bounds = rasterio.features.bounds(shape,transform=None)
            tmp = rev * tuple((bounds[0],bounds[1]))
            x0,y0 = (round(tmp[0]), round(tmp[1])) #left_bottom
            tmp = rev * tuple((bounds[2],bounds[3]))
            x1,y1 = (round(tmp[0]),round(tmp[1])) #right_top
            xmin = np.min([x0,x1])
            xmax = np.max([x0,x1])
            ymin = np.min([y0,y1])
            ymax = np.max([y0,y1])
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax-xmin)*(ymax-ymin))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# ## Load Pytorch Pretrained Model

# In[5]:


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
      
def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=2)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ## Split Train and Validation sets and Create Data Loaders

# In[6]:


from engine import train_one_epoch, evaluate
import utils
import transforms as T

TRAINING_TEST_EXAMPLES = total_files
TEST_EXAMPLES = 100

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# use our dataset and defined transformations
dataset = CircleFinderDataset(directories, get_transform(train=True))
dataset_test = CircleFinderDataset(directories, get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()[:TRAINING_TEST_EXAMPLES]
dataset = torch.utils.data.Subset(dataset, indices[:-TEST_EXAMPLES])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-TEST_EXAMPLES:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)


# ## Choose Hyperparameters

# In[7]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and circle
num_classes = 2

# get the model using our helper function
model = build_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.8, weight_decay=0.00001)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# ## Train Model

# In[8]:


import time
start_time = time.time()
# number of epochs
num_epochs = 7

for epoch in range(num_epochs):
    # train for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

elapsed_time = time.time() - start_time


# In[9]:


hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# In[10]:


torch.save(model, 'circledetectionModel.pt')


# In[ ]:




