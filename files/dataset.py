import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torchvision.transforms.functional as TF
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms

def set_image_dimensions(height, width):
    # fetch the global variables, image_height and image_width from the main script
    global IMAGE_HEIGHT
    global IMAGE_WIDTH
    IMAGE_HEIGHT = height
    IMAGE_WIDTH = width

def set_mean_std(mean, std):
    # fetch the global variables, image_height and image_width from the main script
    global train_mean
    global train_std
    train_mean = mean
    train_std = std

class FranceSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, images, masks, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self.transform((image, mask))

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

def apply_train_transforms(img_mask):
    """
    https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    :param img_mask:
    :return: img, mask
    """
    img, mask = img_mask

    # Resize the image and mask
    img = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))(img)
    mask = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))(mask)

    ##############################
    # Augment image only
    ##############################

    # Add any other custom transforms here
    # Apply color jitter to the image only
    if random.random() < 0.9:
        color_jitter = transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.4, hue=0.1)
        img = color_jitter(img)

    ##############################
    # Augment both image and mask
    ##############################

    # Apply random horizontal and vertical flips
    if random.random() < 0.5:
        hflip = transforms.RandomHorizontalFlip(p=1.0)
        img = hflip(img)
        mask = hflip(mask)

    if random.random() < 0.5:
        vflip = transforms.RandomVerticalFlip(p=1.0)
        img = vflip(img)
        mask = vflip(mask)

    # Apply random rotation and translation (shift)
    # specify hyperparameters for rotation and translation
    ROTATION = 50
    TRANSLATION = 0.45
    # apply transforms
    angle = random.uniform(-ROTATION, ROTATION)
    translate_x = random.uniform(-TRANSLATION * IMAGE_WIDTH, TRANSLATION * IMAGE_WIDTH)
    translate_y = random.uniform(-TRANSLATION * IMAGE_HEIGHT, TRANSLATION * IMAGE_HEIGHT)
    img = TF.affine(img, angle, (translate_x, translate_y), 1, 0)
    mask = TF.affine(mask, angle, (translate_x, translate_y), 1, 0)

    # transform to tensor
    img = transforms.ToTensor()(img)
    mask = transforms.ToTensor()(mask)

    # Normalize the image
    img = transforms.Normalize(mean=train_mean, std=train_std)(img)

    return img, mask

def apply_val_transforms(img_mask):
    """Function to apply the transformations to be used for the validation set.
    Only resizing and converting to tensor.
    No random transformations are applied to the validation set.
    :param img_mask:
    :return: img, mask"""
    img, mask = img_mask

    # Resize the image and mask
    img = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))(img)
    mask = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))(mask)

    # transform to tensor
    img = transforms.ToTensor()(img)
    mask = transforms.ToTensor()(mask)

    # Normalize the image
    img = transforms.Normalize(train_mean, train_std)(img)

    return img, mask

def apply_initial_transforms(img_mask):
    """Function to apply the transformations to be used for the validation set.
    Only resizing and converting to tensor.
    No random transformations are applied to the validation set.
    :param img_mask:
    :return: img, mask"""
    img, mask = img_mask

    # Resize the image and mask
    img = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))(img)
    mask = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))(mask)

    # transform to tensor
    img = transforms.ToTensor()(img)
    mask = transforms.ToTensor()(mask)

    return img, mask

def get_mean_std(train_loader):
    """Function to calculate the mean and standard deviation of the training set.
    :param train_loader:
    :return: mean, std"""
    train_mean = []
    train_std = []

    for batch_idx, (X, y) in enumerate(train_loader):
        numpy_image = X.numpy()

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))

        train_mean.append(batch_mean)
        train_std.append(batch_std)
    train_mean = torch.tensor(np.mean(train_mean, axis=0))
    train_std = torch.tensor(np.mean(train_std, axis=0))
    # convert mean and std to tuple
    print('################# \n For Normalization')
    print('Mean of Training Images:', train_mean)
    print('Std Dev of Training Images:', train_std)
    return train_mean, train_std

class UnNormalize(object):
    # link: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor