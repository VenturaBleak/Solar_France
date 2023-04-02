import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import UNET
import pytorch_warmup as warmup
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
    calculate_binary_metrics
)

def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4 # (0.0001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    if DEVICE == "cuda":
        NUM_WORKERS = 4
    else:
        NUM_WORKERS = 0
    IMAGE_HEIGHT = 416  # 400 originally
    IMAGE_WIDTH = 416  # 400 originally
    PIN_MEMORY = True
    LOAD_MODEL = False

    ############################
    # Script
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

    ############################
    # Train and validation splits
    ############################
    train_images, train_masks, val_images, val_masks = create_train_val_splits(image_dir,
                                                                               mask_dir,
                                                                               val_size=0.1,
                                                                               random_state=RANDOM_SEED)

    ############################
    # Transforms
    ############################
    train_transforms = transforms.Lambda(apply_train_transforms)
    val_transforms = transforms.Lambda(apply_val_transforms)


    ############################
    # Data Loaders
    ############################
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
    num_batches = len(train_loader)

    ############################
    # Model & Loss function
    ############################
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()

    ############################
    # Optimizer
    ############################
    # Adam optimizer
    WEIGHT_DECAY = 1e-4 # (0.0001)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # # Adam" optimizer
    # WEIGHT_DECAY = 1e-2 # (0.01)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # SGD optimizer with momentum and weight decay
    # momentum = 0.9
    # WEIGHT_DECAY = 1e-5
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    ############################
    # Scheduler
    ############################
    # Cosine annealing with warm restarts scheduler
    T_0 =  int(len(train_loader) * (NUM_EPOCHS/20)) # The number of epochs or iterations to complete one cosine annealing cycle.
    T_MULT = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0 = T_0,
                                                                     T_mult=T_MULT,
                                                                     eta_min=LEARNING_RATE * 1e-4,
                                                                     verbose=False)

    # # Polynomial learning rate scheduler
    # MAX_EPOCHS = len(train_loader)
    # POLY_POWER = 2
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer,
    #                                                     max_iter=MAX_EPOCHS,
    #                                                     power=POLY_POWER)

    # ReduceLROnPlateau scheduler
    # FACTOR = 0.2
    # PATIENCE = 10
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                        mode='min',
    #                                                        factor=FACTOR,
    #                                                        patience=PATIENCE,
    #                                                        verbose=True,
    #                                                        min_lr=1e-8)


    ############################
    # Visualize sample images
    ############################
    if DEVICE != "cuda":
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
        exit("Visualized sample images.")

    ############################
    # Training
    ############################

    # create a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    #time all epochs
    start_time = time.time()

    # train the model
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler, device=DEVICE, epoch=epoch)


        # validation
        avg_metrics = calculate_binary_metrics(val_loader, model, device=DEVICE)
        pixel_acc, dice, precision, specificity, recall, f1_score, bg_acc = avg_metrics
        print(f"F1-Score:{f1_score:.3f} | Recall:{recall:.3f} | Precision:{precision:.3f} | Learning Rate:{scheduler.get_last_lr()[0]:.1e}")

        # save model and sample predictions
        if DEVICE == "cuda" and epoch % 5 == 0:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            # save some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE, epoch=epoch)

    print("All epochs completed.")

    #time end
    end_time = time.time()
    # print total training time in hours, minutes, seconds
    print("Total training time: ", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))

if __name__ == "__main__":
    main()