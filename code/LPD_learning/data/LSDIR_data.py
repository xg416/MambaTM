from logging import root
import torch
from torch.utils.data import Dataset
import os, glob, random, cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


class LSDIRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = list(glob.iglob(root_dir + "/*/*.png"))
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

if __name__ == '__main__':
    zernike_dataset = LSDIRDataset(DATASET_PATH, transform=train_transforms)
    # Define the size of the validation set
    val_size = int(len(zernike_dataset) * 0.1)  # 10% for validation
    train_size = len(zernike_dataset) - val_size
    train_dataset, val_dataset = random_split(zernike_dataset, [train_size, val_size])
    # Define the data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )