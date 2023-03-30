import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from dataset import FranceSegmentationDataset,create_train_val_splits
from train import train_fn
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

def main():
    # hyperparameters
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    if DEVICE == "cuda":
        NUM_WORKERS = 2
    else:
        NUM_WORKERS = 0
    IMAGE_HEIGHT = 416  # 400 originally
    IMAGE_WIDTH = 416  # 400 originally
    PIN_MEMORY = True
    LOAD_MODEL = False

    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # Define the image and mask directories under the parent directory
    image_dir = os.path.join(parent_dir, 'data',  'trial', 'images')
    mask_dir = os.path.join(parent_dir, 'data',  'trial', 'masks')

    # Define the train and validation directories under the current working directory
    train_images, train_masks, val_images, val_masks = create_train_val_splits(image_dir,
                                                                               mask_dir,
                                                                               val_size=0.1,
                                                                               random_state=RANDOM_SEED)

    # Define the train transforms, which are applied to the training images and masks
    train_transforms = A.Compose(
        [A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # Define the val transforms, which are applied to the training images and masks
    val_transforms = A.Compose(
        [
            #
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # instantiate model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # get train and validation loaders
    train_loader, val_loader = get_loaders(
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        batch_size=BATCH_SIZE,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # create a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # train the model
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()
    print("success")