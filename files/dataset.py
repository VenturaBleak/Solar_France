import os
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torchvision.transforms.functional as TF
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms

class FranceSegmentationDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, images, masks, transform=None):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        for image_dir in self.image_dirs:
            # specify the path to the image
            image_path = os.path.join(image_dir, self.images[idx])
            # check if the image exists
            if os.path.exists(image_path):
                # if the image exists, break the loop
                break

        # repeat the same process for the mask
        for mask_dir in self.mask_dirs:
            mask_path = os.path.join(mask_dir, self.masks[idx])
            if os.path.exists(mask_path):
                break

        # load the image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # apply the transformations, also pass the image folder path to the apply_train_transforms function
        image, mask = self.transform((image, mask, image_dir))

        return image, mask

def create_train_val_splits(image_dirs, mask_dirs, fractions, val_size=0.2, random_state=42):
    # List all images and masks in the data directories
    images = []
    masks = []
    for image_dir, mask_dir, fraction in zip(image_dirs, mask_dirs, fractions):
        all_images = sorted(os.listdir(image_dir))
        all_masks = sorted(os.listdir(mask_dir))

        num_samples = int(len(all_images) * fraction)

        random.seed(random_state)
        selected_images = random.sample(all_images, num_samples)

        # Find the corresponding masks for the selected images
        selected_masks = [os.path.basename(os.path.join(mask_dir, img_name)) for img_name in selected_images]

        images.extend(selected_images)
        masks.extend(selected_masks)

    # Create a list of tuples, where each tuple contains an image file and its corresponding mask file
    data = list(zip(images, masks))

    # Shuffle the data
    random.seed(random_state)
    random.shuffle(data)

    # Split the data into training and validation subsets
    # exception handling for the case when val_size is >= 1
    if val_size >= 1:
        # then, the val_size is the total number of images
        train_data = data
        val_data = data
    else:
        train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)



    # Separate the images and masks into two separate lists for each subset
    train_images, train_masks = zip(*train_data)
    val_images, val_masks = zip(*val_data)

    # Calculate the total number of images in the training and validation subsets
    total_selected_images = 0
    for image_dir, fraction in zip(image_dirs, fractions):
        images = sorted(os.listdir(image_dir))
        num_images = int(len(images) * fraction)
        total_selected_images += num_images

    return train_images, train_masks, val_images, val_masks, total_selected_images

def get_dirs_and_fractions(dataset_fractions, parent_dir):
    image_dirs = []
    mask_dirs = []
    fractions = []

    for ds_info in dataset_fractions:
        dataset_name, pos_fraction, neg_fraction = ds_info

        image_dirs.append(os.path.join(parent_dir, 'data', dataset_name, 'images_positive'))
        image_dirs.append(os.path.join(parent_dir, 'data', dataset_name, 'images_negative'))

        mask_dirs.append(os.path.join(parent_dir, 'data', dataset_name, 'masks_positive'))
        mask_dirs.append(os.path.join(parent_dir, 'data', dataset_name, 'masks_negative'))

        fractions.append(pos_fraction)
        fractions.append(neg_fraction)

    return image_dirs, mask_dirs, fractions

class TransformationTypes:
    def __init__(self, train_mean, train_std, image_height, image_width, cropping=False):
        self.train_mean = train_mean
        self.train_std = train_std
        self.image_height = image_height
        self.image_width = image_width
        self.cropping = cropping
        self.center_crop_folders = ["Heerlen_2018_HR_output", "Heerlen_2018_HR_output", "Denmark"]

    def center_crop_image(self, img, mask, crop_width, crop_height):
        img_width, img_height = img.size
        mask_width, mask_height = mask.size

        start_x = (img_width - crop_width) // 2
        start_y = (img_height - crop_height) // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        img = img.crop((start_x, start_y, end_x, end_y))
        mask = mask.crop((start_x, start_y, end_x, end_y))

        return img, mask

    def random_crop_image(self, img, mask, crop_width, crop_height):
        img_width, img_height = img.size
        mask_width, mask_height = mask.size

        start_x = random.randint(0, img_width - crop_width)
        start_y = random.randint(0, img_height - crop_height)
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        img = img.crop((start_x, start_y, end_x, end_y))
        mask = mask.crop((start_x, start_y, end_x, end_y))

        return img, mask

    def apply_cropping(self, img, mask, img_folder):
        if self.cropping:
            crop_width, crop_height = 200, 200
            parent_folder = os.path.dirname(img_folder)
            folder_name = os.path.basename(parent_folder)
            if folder_name in self.center_crop_folders:
                img, mask = self.center_crop_image(img, mask, crop_width, crop_height)
            else:
                img, mask = self.random_crop_image(img, mask, crop_width, crop_height)

            img = transforms.Resize((self.image_height, self.image_width))(img)
            mask = transforms.Resize((self.image_height, self.image_width))(mask)

        return img, mask

    def apply_initial_transforms(self, img_mask):
        img, mask, img_folder = img_mask

        # Resize the image and mask
        img = transforms.Resize((self.image_height, self.image_width))(img)
        mask = transforms.Resize((self.image_height, self.image_width))(mask)

        # transform to tensor
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img, mask

    def apply_val_transforms(self, img_mask):
        img, mask, img_folder = img_mask

        # Resize the image and mask
        img = transforms.Resize((self.image_height, self.image_width))(img)
        mask = transforms.Resize((self.image_height, self.image_width))(mask)

        # Apply cropping
        img, mask = self.apply_cropping(img, mask, img_folder)

        # transform to tensor
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        # Normalize the image
        img = transforms.Normalize(self.train_mean, self.train_std)(img)

        return img, mask

    def apply_train_transforms(self, img_mask):
        """
        https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        :param img_mask: tuple containing an image and its corresponding mask
        :param img_folder: the folder containing the image
        :return: img, mask: the transformed image and mask
        """
        img, mask, img_folder = img_mask

        # Resize the image and mask
        img = transforms.Resize((self.image_height, self.image_width))(img)
        mask = transforms.Resize((self.image_height, self.image_width))(mask)

        # Apply cropping
        img, mask = self.apply_cropping(img, mask, img_folder)

        ##############################
        # Augment image only
        ##############################

        # Apply sharpening or smoothing to the
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

        # sharpening
        if random.random() < 0.2:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0, 1))

        # Add any other custom transforms here
        # Apply color jitter to the image only
        if random.random() < 0.8:
            color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.35, saturation=0.25, hue=0.1)
            img = color_jitter(img)

        ##############################
        # Augment both image and mask
        ##############################

        # Zooming in and out by max x%
        ZOOM = 0.1 # 0.1 = 10% -> 10% zoom in or out
        if random.random() < 0.5:
            zoom_factor = random.uniform(1 - ZOOM, 1 + ZOOM)
            zoom_transform = transforms.RandomResizedCrop((self.image_height, self.image_width), scale=(zoom_factor, zoom_factor),
                                                          ratio=(1, 1))
            img = zoom_transform(img)
            mask = zoom_transform(mask)

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
        TRANSLATION = 0.4
        # apply transforms
        angle = random.uniform(-ROTATION, ROTATION)
        translate_x = random.uniform(-TRANSLATION * self.image_width, TRANSLATION * self.image_width)
        translate_y = random.uniform(-TRANSLATION * self.image_height, TRANSLATION * self.image_height)
        img = TF.affine(img, angle, (translate_x, translate_y), 1, 0)
        mask = TF.affine(mask, angle, (translate_x, translate_y), 1, 0)

        # transform to tensor
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        # Normalize the image
        img = transforms.Normalize(mean=self.train_mean, std=self.train_std)(img)

        return img, mask

def get_mean_std(train_loader):
    """Function to calculate the mean and standard deviation of the training set.
    :param train_loader:
    :return: mean, std"""
    print('Retrieving mean and std for Normalization')
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