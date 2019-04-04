import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
import torch
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation, Resize, Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def prepare_data_for_ImageFolder(data_train=['train'], label_name='train.csv'):
    """
    Prepare the data to fullfil the requirements of ImageFolder Class of torchvision.

    """
    # Set the data directory
    home_dir = os.getcwd()
    data_train_folder = data_train
    train_data_dir = os.path.join(home_dir, *data_train_folder)

    label_file_name = label_name
    labels_file_dir = os.path.join(home_dir, data_train_folder[0], label_file_name)
    print(labels_file_dir)
    labels = pd.read_csv(labels_file_dir, header=0)

    # Search if there are lose files
    onlyfiles = [f for f in os.listdir(train_data_dir) if os.path.isfile(os.path.join(train_data_dir, f))]
    if len(onlyfiles) == 0:
        return train_data_dir

    # For each image
    for idx, whale_info in tqdm(labels.iterrows()):
        # Get data
        whale_img, whale_id = whale_info[0], whale_info[1]
        whale_id_dir = os.path.join(train_data_dir, whale_id)

        # Check if directory exists
        if not os.path.exists(whale_id_dir):
            os.makedirs(whale_id_dir)

        # Moves file to the id folder
        file_path = os.path.join(train_data_dir, whale_img)
        shutil.move(file_path, whale_id_dir)

    return train_data_dir

data_train = ['humpback-whale-identification', 'train']
train_data_folder = prepare_data_for_ImageFolder(data_train=data_train)

# Created dataset
train_data_dir = train_data_folder

transforms = Compose([Resize((500, 1000)),
                      RandomCrop((400, 800)),
                      RandomRotation(15),
                      ToTensor(),
                      Normalize((.485, .456, .406), (.229, .224, .225))
])

whale_data = ImageFolder(train_data_dir, transform=transforms)
data_loader = DataLoader(whale_data,
                         batch_size=4,
                         shuffle=True,
                         num_workers=3)

def imshow(img, ax):
    std = np.array([.229, .224, .225]).reshape(3,1,1)
    mean = np.array([.485, .456, .406]).reshape(3,1,1)
    img = img * std + mean # unnormalize
    img = np.clip(img, 0, 1)
    ax.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# Check data
img, label = iter(data_loader).next()
fig, ax = plt.subplots(1,1)
img = img.numpy()[0]
imshow(img, ax)
print(whale_data.classes[label.numpy()[0]])
