import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

class FranceSegmentationDataset(Dataset):
    """
    Custom Dataset class for France's solar panel segmentation. This class only supports traditional
    image loading from provided filepaths.

    Attributes:
        images (list): List of full filepaths to images.
        masks (list): List of full filepaths to masks.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the image, mask and dataset name at the given index.

        Parameters:
            idx (int): The index of the image to return.

        Returns:
            tuple: Tuple containing the image, mask and dataset name.
        """
        # Load the image and mask from the provided filepaths
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        # Extract the dataset name from the image filepath
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(self.images[idx])))

        # Apply the transform if it is provided
        if self.transform:
            image, mask = self.transform((image, mask, dataset_name))

        return image, mask

def get_loaders(
    train_images,
    train_masks,
    val_images,
    val_masks,
    batch_size,
    train_transforms,
    val_transforms,
    num_workers=0,
    pin_memory=True
):
    """
    Create DataLoader objects for training, validation and test datasets.

    :param train_images: List of training image file paths.
    :param train_masks: List of training mask file paths.
    :param val_images: List of validation image file paths.
    :param val_masks: List of validation mask file paths.
    :param batch_size: Number of samples per batch.
    :param train_transforms: Transforms to apply to training data.
    :param val_transforms: Transforms to apply to validation data.
    :param num_workers: Number of worker processes for data loading.
    :param pin_memory: Whether to pin memory for DataLoader.
    :return: DataLoaders for training, validation and test datasets.
    """
    # Create datasets for training, validation, and test
    train_ds = FranceSegmentationDataset(
        images=train_images,
        masks=train_masks,
        transform=train_transforms
    )

    val_ds = FranceSegmentationDataset(
        images=val_images,
        masks=val_masks,
        transform=val_transforms
    )

    # Create data loaders for training, validation, and test
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def fetch_filepaths(image_dirs, mask_dirs, fractions, random_state=42):
    """
    Fetches the filepaths of the images and their corresponding masks from the given directories.

    Parameters:
    image_dirs (list): List of directories containing the images.
    mask_dirs (list): List of directories containing the masks.
    fractions (list): List of fractions specifying how many images to select from each directory.
    random_state (int): The seed for the random number generator.

    Returns:
    list, list: The lists of filepaths for the selected images and their corresponding masks.
    """
    # Initialize lists to store the filepaths of the images and masks
    images = []
    masks = []

    # For each pair of image and mask directories
    for image_dir, mask_dir, fraction in zip(image_dirs, mask_dirs, fractions):
        # List all the images and masks in the current directories
        all_images = sorted(glob.glob(os.path.join(image_dir, '*')))
        all_masks = sorted(glob.glob(os.path.join(mask_dir, '*')))

        # Calculate the number of samples to select based on the given fraction
        num_samples = int(len(all_images) * fraction)

        # Randomly select the images
        random.seed(random_state)
        selected_images = random.sample(all_images, num_samples)

        # Find the corresponding masks for the selected images
        selected_masks = []
        for img_path in selected_images:
            # Extract the image filename
            image_filename = os.path.basename(img_path)

            # Construct the corresponding mask path
            mask_path = os.path.join(mask_dir, image_filename)
            selected_masks.append(mask_path)

        # Add the selected images and masks to the final lists
        images.extend(selected_images)
        masks.extend(selected_masks)

    return images, masks

def get_dirs_and_fractions(dataset_fractions, parent_dir, data_dir='data'):
    """
    Returns the image and mask directories and their corresponding fractions from the given dataset fractions.
    :param dataset_fractions:
    :param parent_dir:
    :param data_dir:
    :return:
    """
    image_dirs = []
    mask_dirs = []
    fractions = []

    for ds_info in dataset_fractions:
        dataset_name, pos_fraction, neg_fraction = ds_info

        image_dirs.append(os.path.join(parent_dir, data_dir, dataset_name, 'images_positive'))
        image_dirs.append(os.path.join(parent_dir, data_dir, dataset_name, 'images_negative'))

        mask_dirs.append(os.path.join(parent_dir, data_dir, dataset_name, 'masks_positive'))
        mask_dirs.append(os.path.join(parent_dir, data_dir, dataset_name, 'masks_negative'))

        fractions.append(pos_fraction)
        fractions.append(neg_fraction)

    return image_dirs, mask_dirs, fractions