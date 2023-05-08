import os
import random
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from utils import check_non_binary_pixels

class TransformationTypes:
    def __init__(self, train_mean, train_std, image_height, image_width, cropping=False):
        self.train_mean = train_mean
        self.train_std = train_std
        self.image_height = image_height
        self.image_width = image_width
        self.cropping = cropping
        self.center_crop_folders = ["Heerlen_2018_HR_output", "Heerlen_2018_HR_output", "Denmark", "China"]

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
            crop_width, crop_height = 256, 256
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
        ZOOM = 0.2  # 0.1 = 10% -> 10% zoom in or out
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