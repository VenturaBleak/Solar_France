import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split

class FranceSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, mask_list, transform=None):
        # image_dir: path to the image directory
        self.image_dir = image_dir
        # mask_dir: path to the mask directory
        self.mask_dir = mask_dir
        # transform: transform to apply to the image and mask
        self.transform = transform

        # list all images and masks in the data directory
        self.images = image_list
        self.masks = mask_list

    def __len__(self):
        # return the length of the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # load the image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # convert image and mask to numpy array
        # image to RGB
        image = np.array(Image.open(image_path).convert("RGB"))
        # mask to grayscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # convert the mask to a binary mask, where 1 is the foreground and 0 is the background
        mask[mask > 0] = 1.0

        if self.transform is not None:
            # apply transformations, store the augmented image and mask in a dictionary called augmented
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

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