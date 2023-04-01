import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import UNET
from dataset import (FranceSegmentationDataset,
                     create_train_val_splits,
                     apply_train_transforms,
                     apply_val_transforms,
                     set_image_dimensions)
from train import train_fn
from data_cleaning import remove_unmatched_files
from image_size_check import check_dimensions
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    if DEVICE == "cuda":
        NUM_WORKERS = 4
    else:
        NUM_WORKERS = 0
    IMAGE_HEIGHT = 416  # 400 originally
    IMAGE_WIDTH = 416  # 400 originally
    PIN_MEMORY = True
    LOAD_MODEL = False

    ############################
    # script
    ############################
    # pass on the image dimensions to the dataset class
    set_image_dimensions(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    if DEVICE == "cuda":
        image_dir = os.path.join(parent_dir, 'data', 'bdappv', 'google', 'images')
        mask_dir = os.path.join(parent_dir, 'data', 'bdappv', 'google', 'masks')

    else:
        image_dir = os.path.join(parent_dir, 'data', 'trial', 'images')
        mask_dir = os.path.join(parent_dir, 'data', 'trial', 'masks')

    # remove unmatched images and masks
    remove_unmatched_files(image_dir, mask_dir)

    # assert that the number of images and masks are equal
    assert len(os.listdir(image_dir)) == len(os.listdir(mask_dir))

    # assert that dimensions of each image are equal
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    check_dimensions(image_dir, mask_dir, images, masks)


    # Define the train and validation directories under the current working directory
    train_images, train_masks, val_images, val_masks = create_train_val_splits(image_dir,
                                                                               mask_dir,
                                                                               val_size=0.1,
                                                                               random_state=RANDOM_SEED)

    train_transforms = transforms.Lambda(apply_train_transforms)
    val_transforms = transforms.Lambda(apply_val_transforms)


    ############################
    # Model & Optimizer
    ############################
    # instantiate model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    ############################
    # Scheduler
    ############################
    T_max = int(num_epochs/4) # The number of epochs or iterations to complete one cosine annealing cycle.
    eta_min = 1e-7 # The minimum learning rate at the end of each cycle
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=T_max,
                                                           eta_min=eta_min)

    ############################
    # Data Loaders
    ############################
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

    ############################
    # Visualize sample images
    ############################

    # visualize some sample images
    import matplotlib.pyplot as plt
    import numpy as np

    images, masks = next(iter(train_loader))
    n_samples = BATCH_SIZE // 2

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(n_samples, 2, figsize=(8, n_samples * 4))

    # Iterate over the images and masks and plot them side by side
    for i in range(n_samples):
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        mask = np.squeeze(masks[i].numpy(), axis=0)

        axs[i, 0].axis("off")
        axs[i, 0].imshow(img)
        axs[i, 1].axis("off")
        axs[i, 1].imshow(mask, cmap="gray")
    plt.show()
    # exit()

    ############################
    # Training
    ############################

    # create a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # train the model
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler, device=DEVICE)

        if DEVICE == "cuda":
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()
    print("success")