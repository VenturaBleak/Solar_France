import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
import pandas as pd
from model import (UNET,
                   Segformer, create_segformer,
                   DiceLoss, DiceBCELoss, FocalLoss, IoULoss, TverskyLoss,
                   PolynomialLRDecay, GradualWarmupScheduler
                   )
from dataset import (FranceSegmentationDataset, TransformationTypes,
                     create_train_val_splits,get_dirs_and_fractions,
                     get_mean_std, UnNormalize)
from train import train_fn
from image_size_check import check_dimensions
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
    calculate_binary_metrics,
    generate_model_name
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
    WARMUP_EPOCHS = int(NUM_EPOCHS * 0.05) # 5% of the total epochs
    CROPPING = True

    ############################
    # Script
    ############################
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # specify the training datasets
    if DEVICE == "cuda":
        dataset_fractions = [
        # [dataset_name, fraction_of_positivies, fraction_of_negatives]
            ['France_google', 1, 1],
            ['Munich', 1, 0],
            ['Heerlen_2018_HR_output', 0, 0.1],
            ['Denmark', 0, 0.1]
        ]
    else:
        dataset_fractions = [
        # [dataset_name, fraction_of_positivies, fraction_of_negatives]
            ['France_google', 1, 0.0],
            ['Munich', 0.0, 0.0],
            ['Denmark', 0.0, 0.0],
            ['Heerlen_2018_HR_output', 0.0, 0.0],
            ['ZL_2018_HR_output', 0.0, 0.0]
        ]

    image_dirs, mask_dirs, fractions = get_dirs_and_fractions(dataset_fractions, parent_dir)

    ############################
    # Train and validation splits
    ############################
    # Train and validation splits
    train_images, train_masks, val_images, val_masks, total_selected_images = create_train_val_splits(
        image_dirs,
        mask_dirs,
        fractions,
        val_size=0.1,
        random_state=RANDOM_SEED,
    )

    # assert that images and masks are identical
    assert train_images == train_masks
    assert val_images == val_masks


    ############################
    # Specify initial transforms
    ############################
    transformations = TransformationTypes(None, None, IMAGE_HEIGHT, IMAGE_WIDTH, cropping=False)
    inital_transforms = transforms.Lambda(transformations.apply_initial_transforms)

    ############################
    # Check data loading is correct
    ############################

    # assert that the number of images and masks are the same
    train_ds = FranceSegmentationDataset(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        images=train_images,
        masks=train_masks,
        transform=inital_transforms,
    )
    val_ds = FranceSegmentationDataset(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        images=val_images,
        masks=val_masks,
        transform=inital_transforms,
    )
    assert (len(train_ds) + len(val_ds)) == total_selected_images
    print(f"Total number of images: {total_selected_images}, "
          f"of which {len(train_ds)} are training images and {len(val_ds)} are validation images.")
    del val_ds, train_ds

    ############################
    # Get mean and std of training set
    ############################
    # Get loaders
    train_loader, val_loader = get_loaders(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        batch_size=BATCH_SIZE,
        train_transforms=inital_transforms,
        val_transforms=inital_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    # retrieve the mean and std of the training images
    train_mean, train_std = get_mean_std(train_loader)

    ############################
    # Transforms
    ############################
    transformations = TransformationTypes(train_mean, train_std, IMAGE_HEIGHT, IMAGE_WIDTH, cropping=CROPPING)
    train_transforms = transforms.Lambda(transformations.apply_train_transforms)
    val_transforms = transforms.Lambda(transformations.apply_val_transforms)

    ############################
    # Data Loaders
    ############################
    # Get loaders
    train_loader, val_loader = get_loaders(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
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
    # UNET
    # model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # Segformer
    if DEVICE == "cuda":
        segformer_arch = 'B0'
    else:
        segformer_arch = 'B0'
    # also, experiment with bilinear interpolation on or off -> final upsamplin layer of the model
    model = create_segformer(segformer_arch, channels=3, num_classes=1).to(DEVICE)

    # model summary
    summary(model, input_size=(BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    ############################
    # Loss function
    ############################
    # BCE
    # loss_fn = nn.BCEWithLogitsLoss()

    # Dice
    # loss_fn = DiceLoss()

    # IoU
    # loss_fn = IoULoss()

    # Tversky
    # loss_fn = TverskyLoss()

    # Careful: Loss functions below do not work with autocast in training loop!
    # Dice + BCE
    loss_fn = DiceBCELoss()

    # Focal
    # loss_fn = FocalLoss()

    ############################
    # Optimizer
    ############################
    # Dynamic weight decay
    # link https://discuss.pytorch.org/t/change-weight-decay-during-training/70377/2

    # Adam optimizer
    # WEIGHT_DECAY = 1e-5 # (0.00001)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # AdamW optimizer
    WEIGHT_DECAY = 1e-2 # (0.01)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # SGD optimizer with momentum and weight decay
    # momentum = 0.9
    # WEIGHT_DECAY = 1e-5
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    ############################
    # Scheduler
    ############################
    # update the learning rate after each batch for the following schedulers
    # Cosine annealing with warm restarts scheduler
    T_0 =  int((len(train_loader) * NUM_EPOCHS - (len(train_loader) * WARMUP_EPOCHS))/30) # The number of epochs or iterations to complete one cosine annealing cycle.
    print('Cosing Annealing with Warm Restarts scheduler - Number of batches in T_0:', T_0)
    T_MULT = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0 = T_0,
                                                                     T_mult=T_MULT,
                                                                     eta_min=LEARNING_RATE * 1e-4,
                                                                     verbose=False)

    # update the learning rate after each epoch for the following schedulers
    # Polynomial learning rate scheduler
    # scheduler visualized: https://www.researchgate.net/publication/224312922/figure/fig1/AS:668980725440524@1536508842675/Plot-of-Q-1-of-our-upper-bound-B1-as-a-function-of-the-decay-rate-g-for-both.png
    # MAX_ITER = NUM_EPOCHS - WARMUP_EPOCHS
    # print('Polynomial learning rate scheduler - MAX_Iter (number of iterations until decay):', MAX_ITER)
    # POLY_POWER = 2.0
    # scheduler = PolynomialLRDecay(optimizer=optimizer,
    #                               max_decay_steps=MAX_ITER, # when to stop decay
    #                               end_learning_rate=LEARNING_RATE*1e-4,
    #                               power=POLY_POWER)

    # Scheduler warmup
    # handle batch level schedulers
    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)\
            or isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        WARMUP_EPOCHS = WARMUP_EPOCHS * len(train_loader)
        print("Number of Warmup Batches: ", WARMUP_EPOCHS)
        IS_BATCH = True
    else:
        IS_BATCH = False
        print("Number of Warmup Epochs: ", WARMUP_EPOCHS)

    # GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=WARMUP_EPOCHS, # when to stop warmup
                                       after_scheduler=scheduler,
                                       is_batch = IS_BATCH)

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

    # unnormalize
    unorm = UnNormalize(mean=tuple(train_mean.numpy()), std=(tuple(train_std.numpy())))

    # Iterate over the images and masks and plot them side by side
    for i in range(n_samples):
        img = unorm(images[i].squeeze(0))
        img = np.transpose(img.numpy(), (1, 2, 0))
        mask = np.squeeze(masks[i].numpy(), axis=0)

        axs[i, 0].axis("off")
        axs[i, 0].imshow(img)
        axs[i, 1].axis("off")
        axs[i, 1].imshow(mask, cmap="gray")
    plt.show()
    # exit("Visualized sample images.")

    ############################
    # Training
    ############################

    # retrieve model name for saving
    model_name = generate_model_name(segformer_arch,
                                     loss_fn.__class__.__name__,
                                     optimizer.__class__.__name__,
                                     [x[0] for x in dataset_fractions])

    # create a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # Time total training time
    start_time = time.time()

    # Initialize an empty DataFrame to store the logs
    log_df = pd.DataFrame(
        columns=["epoch", "learning_rate", "train_loss", "val_loss", "val_f1-score", "val_precision", "val_recall",
                 "val_pixel_acc", "val_iou", "val_specificity", "val_bg-acc"])

    # Initialize the best validation metric
    best_val_metric = float('-inf')  # Use float('inf') for loss, or float('-inf') for F1-score and other metrics

    # train the model
    for epoch in range(NUM_EPOCHS):
        # Train
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler, device=DEVICE, epoch=epoch)

        # Validate
        avg_metrics = calculate_binary_metrics(val_loader, model, loss_fn, device=DEVICE)
        pixel_acc, iou, precision, specificity, recall, f1_score, bg_acc, val_loss = avg_metrics
        print(
            f"Val.Metrics: F1-Score:{f1_score:.3f} | Recall:{recall:.3f} | Precision:{precision:.3f} | Loss: {val_loss:.4f} | LR:{scheduler.get_last_lr()[0]:.1e}")

        # Log & save the validation metrics
        log_df.loc[epoch] = [epoch, scheduler.get_last_lr()[0], epoch_loss, val_loss, f1_score, precision, recall,
                             pixel_acc, iou, specificity, bg_acc]

        # Saving the model, if the current validation metric is better than the best one
        if f1_score > best_val_metric:  # Change the condition if using val_loss or another metric
            best_val_metric = f1_score

            # save model and sample predictions
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            model_path = save_checkpoint(checkpoint, model_name=model_name, parent_dir=parent_dir)
            # save some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, unnorm=unorm, model_name=model_name, folder=model_path,
                device=DEVICE, testing=False)

        # Save the logs
        log_csv_path = os.path.join(model_path, f"{model_name}_logs.csv")
        log_df.to_csv(log_csv_path, index=False)

    print("All epochs completed.")

    #time end
    end_time = time.time()

    # print total training time in hours, minutes, seconds
    print("Total training time: ", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))

if __name__ == "__main__":
    main()