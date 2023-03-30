import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import random
from sklearn.model_selection import train_test_split

class FranceSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, images, masks, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = images
        self.masks = masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert the mask to binary
        mask_np = np.array(mask, dtype=np.float32)
        mask_np[mask_np > 0] = 1.0

        # Apply transformations
        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask)

        return image_tensor, mask_tensor

def create_train_val_splits(image_dir, mask_dir, val_size=0.2, random_state=42):
    # List all images and masks in the data directory
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))

    # Create a list of tuples, where each tuple contains an image file and its corresponding mask file
    data = list(zip(images, masks))

    # Shuffle the data
    random.seed(random_state)
    random.shuffle(data)

    # Split the data into training and validation subsets
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)

    # Separate the images and masks into two separate lists for each subset
    train_images, train_masks = zip(*train_data)
    val_images, val_masks = zip(*val_data)

    return train_images, train_masks, val_images, val_masks