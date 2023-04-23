import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

class FranceSegmentationDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, images, masks, transform=None, image_gen_func=None, extra_images=0):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.images = images
        self.masks = masks
        self.transform = transform
        self.image_gen_func = image_gen_func
        self.extra_images = extra_images

    def __len__(self):
        return len(self.images) + self.extra_images

    def __getitem__(self, idx):
        if idx < len(self.images):
            # Use the traditional method for loading images and masks
            for image_dir in self.image_dirs:
                image_path = os.path.join(image_dir, self.images[idx])
                if os.path.exists(image_path):
                    break

            for mask_dir in self.mask_dirs:
                mask_path = os.path.join(mask_dir, self.masks[idx])
                if os.path.exists(mask_path):
                    break

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

        else:
            # Call the image generation function for additional images
            image, mask = self.image_gen_func()
            image_dir = "snippet"

        image, mask = self.transform((image, mask, image_dir))
        return image, mask, self.images[idx] if idx < len(self.images) else f"generated_{idx - len(self.images)}"

def get_loaders(
    image_dirs,
    mask_dirs,
    train_images,
    train_masks,
    val_images,
    val_masks,
    batch_size,
    train_transforms,
    val_transforms,
    num_workers=0,
    pin_memory=True,
    image_gen_func=None,
    extra_images=0,
    train_sampler = None,
    random_seed = 42
):

    train_ds = FranceSegmentationDataset(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        images=train_images,
        masks=train_masks,
        transform=train_transforms,
        image_gen_func=image_gen_func,
        extra_images=extra_images
    )

    if train_sampler is not None:
        train_sampler = CustomSampler(len(train_ds.images), train_ds.extra_images)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True
    )

    val_ds = FranceSegmentationDataset(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        images=val_images,
        masks=val_masks,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

class CustomSampler(Sampler):
    def __init__(self, num_original, num_additional, generator=None, random_seed=42):
        self.num_original = num_original
        self.num_additional = num_additional
        self.generator = generator or torch.Generator().manual_seed(random_seed)

    def __iter__(self):
        original_indices = torch.arange(self.num_original)
        additional_indices = torch.arange(self.num_original, self.num_original + self.num_additional)
        all_indices = torch.cat([original_indices, additional_indices])
        shuffled_indices = torch.randperm(len(all_indices), generator=self.generator)
        return iter(all_indices[shuffled_indices])

    def __len__(self):
        return self.num_original + self.num_additional

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

def filter_positive_images(train_images, train_masks, image_dirs, mask_dirs):
    train_images_positive = []
    train_masks_positive = []
    positive_image_dirs = []
    positive_mask_dirs = []

    for image, mask in zip(train_images, train_masks):
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            image_path = os.path.join(image_dir, image)
            if os.path.exists(image_path) and 'images_positive' in image_dir:
                train_images_positive.append(image)
                train_masks_positive.append(mask)
                if image_dir not in positive_image_dirs:
                    positive_image_dirs.append(image_dir)
                if mask_dir not in positive_mask_dirs:
                    positive_mask_dirs.append(mask_dir)

    return train_images_positive, train_masks_positive, positive_image_dirs, positive_mask_dirs

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

            img = transforms.Resize((self.image_height, self.image_width),
                                    interpolation=transforms.InterpolationMode.BICUBIC)(img)
            mask = transforms.Resize((self.image_height, self.image_width),
                                     interpolation=transforms.InterpolationMode.NEAREST)(mask)

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
        img = transforms.Resize((self.image_height, self.image_width), interpolation = transforms.InterpolationMode.BICUBIC)(img)
        mask = transforms.Resize((self.image_height, self.image_width), interpolation = transforms.InterpolationMode.NEAREST)(mask)
        # uncomment to assert that the mask is binary
        # check_non_binary_pixels(mask, "resize")

        # Apply cropping
        img, mask = self.apply_cropping(img, mask, img_folder)

        ##############################
        # Augment image only
        ##############################

        # Apply sharpening or smoothing to the
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

        # sharpening
        if random.random() < 0.3:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0, 1))

        # Add any other custom transforms here
        # Apply color jitter to the image only
        if random.random() < 0.8:
            color_jitter = transforms.ColorJitter(brightness=0.45, contrast=0.35, saturation=0.25, hue=0.1)
            img = color_jitter(img)

        ##############################
        # Augment both image and mask
        ##############################

        # Zooming in and out by max x%
        ZOOM = 0.1  # 0.1 = 10% -> 10% zoom in or out
        PADDING = int(max(self.image_height, self.image_width) * ZOOM)
        if random.random() < 0.8:
            # Resize the image and mask with some padding
            img = transforms.Resize((self.image_height + PADDING, self.image_width + PADDING),
                                    interpolation=transforms.InterpolationMode.BICUBIC)(img)
            mask = transforms.Resize((self.image_height + PADDING, self.image_width + PADDING),
                                     interpolation=transforms.InterpolationMode.NEAREST)(mask)

            # Apply the same random crop to both the image and the mask
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.image_height, self.image_width))
            img = transforms.functional.crop(img, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
            # uncomment to assert that the mask is binary
            # check_non_binary_pixels(mask, "zoom")

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

        # Generate random parameters for the affine transformation
        angle = random.uniform(-ROTATION, ROTATION)
        translate_x = random.uniform(-TRANSLATION * self.image_width, TRANSLATION * self.image_width)
        translate_y = random.uniform(-TRANSLATION * self.image_height, TRANSLATION * self.image_height)

        # Apply the affine transformation to the image with the same parameters
        img = transforms.functional.affine(img, angle, (translate_x, translate_y), 1, 0,
                                           interpolation=transforms.InterpolationMode.BICUBIC)

        # Apply the affine transformation to the mask with the same parameters
        mask = transforms.functional.affine(mask, angle, (translate_x, translate_y), 1, 0,
                                            interpolation=transforms.InterpolationMode.NEAREST)
        # uncomment to assert that the mask is binary
        #check_non_binary_pixels(mask, "affine")

        # transform to tensor
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        # Normalize the image
        img = transforms.Normalize(mean=self.train_mean, std=self.train_std)(img)

        return img, mask

def check_non_binary_pixels(mask,transformation):
    tensor_mask = transforms.ToTensor()(mask)
    unique_values = torch.unique(tensor_mask)
    for value in unique_values:
        assert value == 0 or value == 1, f"After transformation: {transformation}; Mask contains non-binary value: {value}"

def get_mean_std(train_loader):
    """Function to calculate the mean and standard deviation of the training set.
    :param train_loader:
    :return: mean, std"""
    print('Retrieving mean and std for Normalization')
    train_mean = []
    train_std = []

    for batch_idx, (X, y, _) in enumerate(train_loader):
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