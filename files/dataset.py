import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

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
        return image, mask, self.images[idx] if idx < len(self.images) else f"generated_{idx - len(self.images)}", image_dir

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