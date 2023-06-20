# libraries
import os
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# repo files
from model import (UNET,
                   Segformer, create_segformer
                   )
from dataset import (FranceSegmentationDataset, get_loaders,
                     fetch_filepaths, get_dirs_and_fractions
                     )
from transformations import (TransformationTypes)
from loss_functions import (DiceLoss, DiceBCELoss, FocalLoss, IoULoss, TverskyLoss)
from lr_schedulers import (PolynomialLRDecay, GradualWarmupScheduler)
from train import train_fn
from image_size_check import check_dimensions
from utils import (
    save_checkpoint, load_model, generate_model_name,
    visualize_sample_images, save_predictions_as_imgs, create_gif_from_images,
    count_samples_in_loader,
    update_log_df,
    UnNormalize,
    get_mean_std
)
from eval_metrics import (BinaryMetrics)
from solar_snippet_v2 import ImageProcessor
from feature_maps import visualize_feature_maps
from grad_cam import visualize_gradcam_UNET

def main(model_arch):
    plt.ioff()
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 122204
    LEARNING_RATE = 1e-4 # (0.0001)
    scheduler_name = "PolynomialLRDecay"
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
    CROPPING = False
    CALCULATE_MEAN_STD = False

    ############################
    # set seeds
    ############################
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = False

    ###################################################################################################################
    #### Specify Training, Validation and Test Datasets
    ###################################################################################################################
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    ############################
    # Training dataset
    ############################

    # specify the training datasets
    # train_folder = 'data_snippet'
    # train_ds = [
    # # [dataset_name, fraction_of_positivies, fraction_of_negatives]
    #     ['Munich_TA', 0, 0],
    #     ['Munich_RA', 0, 0],
    #     ['Heerlen_RA', 1, 0],
    #     ['ZL_RA', 1, 0],
    # ]

    train_folder = 'data_train'
    train_ds = [
    # [dataset_name, fraction_of_positivies, fraction_of_negatives]
        ['France_google', 0, 0],
        ['France_ign', 0, 0],
        ['Munich', 0, 0],
        ['China', 0, 0],
        ['Denmark', 0, 0],
        ['Heerlen_2018_HR_output', 1, 0],
        ['ZL_2018_HR_output', 1, 0],
    ]

    image_dirs, mask_dirs, fractions = get_dirs_and_fractions(train_ds, parent_dir, train_folder)
    train_images, train_masks = fetch_filepaths(
        image_dirs,
        mask_dirs,
        fractions,
        random_state=RANDOM_SEED,
    )

    ############################
    # Test dataset
    ############################

    # specify the validation datasets
    val_folder = 'data_test'

    val_ds = [
        # [dataset_name, fraction_of_positivies, fraction_of_negatives]
        ['France_google', 0, 0],
        ['France_ign', 0., 0],
        ['Munich', 0, 0],
        ['China', 0., 0],
        ['Denmark', 0., 0],
        ['Heerlen_2018_HR_output', 1, 0],
        ['ZL_2018_HR_output', 1, 0],
    ]

    # get all images in a given folder, that is: val_data
    val_image_dirs, val_mask_dirs, val_fractions = get_dirs_and_fractions(val_ds, parent_dir, val_folder)
    val_images, val_masks = fetch_filepaths(
        val_image_dirs,
        val_mask_dirs,
        val_fractions,
        random_state=RANDOM_SEED,
    )

    ############################
    # Visualization dataset
    ############################

    # specify the vis datasets
    vis_folder = 'data_test'

    vis_ds = [
        # [dataset_name, fraction_of_positivies, fraction_of_negatives]
        ['France_google', 0., 0],
        ['France_ign', 0., 0],
        ['Munich', 0, 0],
        ['China', 0., 0],
        ['Denmark', 0., 0],
        ['Heerlen_2018_HR_output', 1, 0],
        ['ZL_2018_HR_output', 1, 0],
    ]

    # get all images in a given folder, that is: val_data
    vis_image_dirs, vis_mask_dirs, vis_fractions = get_dirs_and_fractions(vis_ds, parent_dir, vis_folder)
    vis_images, vis_masks = fetch_filepaths(
        vis_image_dirs,
        vis_mask_dirs,
        vis_fractions,
        random_state=RANDOM_SEED,
    )

    ############################
    # Unit Tests for checking filepaths are correctly fetched
    ############################
    # Unit Test1: check whether images are unique, that is, no duplicates
    assert len(train_images) == len(set(train_images))
    assert len(train_masks) == len(set(train_masks))
    assert len(val_images) == len(set(val_images))
    assert len(val_masks) == len(set(val_masks))

    # Unit Test2: assert that the last part of the path is identical for images and masks, for both train and val
    for img_path, mask_path in zip(train_images, train_masks):
        img_parts = img_path.split(os.sep)
        mask_parts = mask_path.split(os.sep)
        # "data\\France_google\\masks_positive\\UMDRQB0BCRQMH.png" -> "France_google" == "France_google"
        assert img_parts[-3:-2] == mask_parts[-3:-2], "Mismatch between image and mask folders"
        # "data\\France_google\\masks_positive\\UMDRQB0BCRQMH.png" -> "UMDRQB0BCRQMH.png" == "UMDRQB0BCRQMH.png"
        assert img_parts[-1].split('.')[0] == mask_parts[-1].split('.')[0], "Mismatch between image and mask filenames"

    # Unit test3 Check for potential mix-up due to identical filenames in different datasets
    unique_image_names = set()
    for dataset, img_list in [('train', train_images), ('val', val_images)]:
        for img_path in img_list:
            dataset_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # e.g. "France_google"
            image_filename = os.path.basename(img_path)  # e.g. "UMDRQB0BCRQMH.png"
            unique_identifier = f"{dataset_name}_{dataset}_{image_filename}"  # e.g. "France_google_train_UMDRQB0BCRQMH.png"

            # Assert that the combination of dataset name, subset (train or val) and image filename is unique
            assert unique_identifier not in unique_image_names, "Potential mix-up due to identical filenames in different datasets"
            unique_image_names.add(unique_identifier)

    ############################
    # Specify initial transforms
    ############################
    transformations = TransformationTypes(None, None, IMAGE_HEIGHT, IMAGE_WIDTH, cropping=False)
    inital_transforms = transforms.Lambda(transformations.apply_initial_transforms)

    ############################
    # Get mean and std of training set
    ############################
    if CALCULATE_MEAN_STD == True:
        # Get loaders
        train_loader, val_loader = get_loaders(
                train_images=train_images,
                train_masks=train_masks,
                val_images=val_images,
                val_masks=val_masks,
                batch_size=BATCH_SIZE,
                train_transforms=inital_transforms,
                val_transforms=inital_transforms,
                num_workers=NUM_WORKERS,
                pin_memory=True
        )
        # retrieve the mean and std of the training images
        train_mean, train_std = get_mean_std(train_loader)
    else:
        # specify train_mean, train_std -> has to be in this format: tensor([0.2929, 0.2955, 0.2725]), tensor([0.2268, 0.2192, 0.2098])
        train_mean = torch.tensor([0.3637, 0.3688, 0.3301])
        train_std = torch.tensor([0.2242, 0.2102, 0.2033])

    # specify UnNormalize() function for visualization of sample images
    unorm = UnNormalize(mean=tuple(train_mean.numpy()), std=(tuple(train_std.numpy())))

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
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        batch_size=BATCH_SIZE,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    _, vis_loader = get_loaders(
        train_images=train_images,
        train_masks=train_masks,
        val_images=vis_images,
        val_masks=vis_masks,
        batch_size=BATCH_SIZE,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    ############################
    # Model & Loss function
    ############################
    if model_arch == "UNet":
        # UNET
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    else:
        # also, experiment with bilinear interpolation on or off -> final upsampling layer of the model
        model = create_segformer(model_arch, channels=3, num_classes=1).to(DEVICE)

    # model summary
    summary(model, input_size=(BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    ############################
    # Loss function
    ############################
    # BCE
    #loss_fn = nn.BCEWithLogitsLoss()

    # Dice
    # loss_fn = DiceLoss()

    # IoU
    # loss_fn = IoULoss()

    # Tversky
    loss_fn = TverskyLoss()

    # Careful: Loss functions below do not work with autocast in training loop!
    # Dice + BCE
    # loss_fn = DiceBCELoss()

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
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    ############################
    # LR Scheduler
    ############################
    # update the learning rate after each batch for the following schedulers
    # Cosine annealing with warm restarts scheduler
    if scheduler_name == "CosineAnnealingWarmRestarts":
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
    if scheduler_name == "PolynomialLRDecay":
        MAX_ITER = int(len(train_loader) * NUM_EPOCHS - (len(train_loader) * WARMUP_EPOCHS))
        print('Polynomial learning rate scheduler - MAX_Iter (number of iterations until decay):', MAX_ITER)
        POLY_POWER = 1.3 # specify the power of the polynomial, 1.0 means linear decay, and 2.0 means quadratic decay
        scheduler = PolynomialLRDecay(optimizer=optimizer,
                                      max_decay_steps=MAX_ITER, # when to stop decay
                                      end_learning_rate=LEARNING_RATE*1e-3,
                                      power=POLY_POWER)

    # LR Scheduler warmup
    if True:
        # applicable for CosineAnnealingWarmRestarts, and PolynomialLRDecay
        WARMUP_EPOCHS = WARMUP_EPOCHS * len(train_loader)
        print("Number of Warmup Batches: ", WARMUP_EPOCHS)
        print(f'Number of total Batches: {len(train_loader) * NUM_EPOCHS}')
        IS_BATCH = True
    else:
        IS_BATCH = False
        print("Number of Warmup Epochs: ", WARMUP_EPOCHS)
        print(f'Number of total Epochs: {NUM_EPOCHS}')

    # GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=WARMUP_EPOCHS, # when to stop warmup
                                       after_scheduler=scheduler,
                                       is_batch = IS_BATCH)

    ############################
    # Loading initialized model weights
    ############################
    model_dir = "_Initialized"

    #ToDo: delete this
    model_dir = "Experiment4"
    model_arch = "UNet_NL_BCE"

    try:
        # load the model
        load_model(model_dir, model_arch, model, parent_dir)
        print(f"Initialized Model {model_arch} loaded.")
    except FileNotFoundError:
        # if model is not found, save the initial state of the model
        print(f"No initialized model with name {model_arch} found, saving the initial state of the model.")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        model_path = save_checkpoint(checkpoint, model_dir=model_dir, model_name=model_arch, parent_dir=parent_dir)

    # ToDo: delete this
    model_arch = "UNet"

    ############################
    # Visualize sample images
    ############################
    # visualize some sample images
    visualize_sample_images(train_loader, train_mean, train_std, BATCH_SIZE, unorm)

    # Print the number of samples in the train and validation loaders
    print(f'Training samples: {len(train_images)} | Training batches: {len(train_loader)}')
    print(f'Validation samples: {len(val_images)} | Validation batches: {len(val_loader)}')

    ############################
    # Training
    ############################

    # retrieve model name for saving
    model_dir = "Experiment4"
    model_name = model_arch + "_" + "NL_Tvserky_ST1"

    # create a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # Time total training time
    start_time = time.time()

    # instantiate BinaryMetrics class & create a dataframe to store the training metrics
    binary_metrics = BinaryMetrics()
    metrics_names = ["epoch", "learning_rate", "train_loss"]
    metrics_names.extend(binary_metrics.metrics.keys())
    log_df = pd.DataFrame(columns=metrics_names)

    # Initialize the best validation metric
    best_val_metric = float('-inf')  # Use float('inf') for loss, or float('-inf') for F1-score and other metrics

    # save model and sample predictions
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    model_path = save_checkpoint(checkpoint, model_dir=model_dir, model_name=model_name, parent_dir=parent_dir)

    # train the model
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler, device=DEVICE, epoch=epoch,
                              model_path=model_path, model_name=model_name)

        # Validate
        metric_dict = binary_metrics.calculate_binary_metrics(val_loader, model, loss_fn, device=DEVICE)
        metrics_names.extend(metric_dict.keys())

        # Print the validation metrics
        print(
            f"Val.Metrics: Loss: {metric_dict['val_loss']:.4f} | "
            f"F1-Score:{metric_dict['f1_score']:.3f} | Precision:{metric_dict['precision']:.3f} | "
            f"Recall:{metric_dict['recall']:.3f} | Balanced-Acc:{metric_dict['balanced_acc']:.3f} |"
            f" LR:{scheduler.get_last_lr()[0]:.1e}"
        )

        # Log validation metrics in a df, in the same order as the metrics_names
        log_df = update_log_df(log_df, metric_dict, epoch, train_loss, scheduler)

        current_val_metric = metric_dict['f1_score']
        # Saving the model, if the current validation metric is better than the best one
        if current_val_metric > best_val_metric:  # Change the condition if using val_loss or another metric
            best_val_metric = current_val_metric

            # save model and sample predictions
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, model_dir=model_dir, model_name=model_name, parent_dir=parent_dir)

            # save some examples to a folder
            save_predictions_as_imgs(
                vis_loader, model, unnorm=unorm, model_name=model_name, folder=model_path,
                device=DEVICE, BATCH_SIZE=BATCH_SIZE)

        # Save the logs
        log_csv_path = os.path.join(model_path, f"{model_name}_logs.csv")
        log_df.to_csv(log_csv_path, index=False)

        # if epoch // 5 == 0: then save pred as imgs
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            # save some examples to a folder
            imgs_file_name = model_name + "_Epoch" + str(epoch)
            save_predictions_as_imgs(
                vis_loader, model, unnorm=unorm, model_name=imgs_file_name, folder=model_path,
                device=DEVICE, BATCH_SIZE=BATCH_SIZE)

            # save feature maps, if UNet
            if model_arch == "UNet":
                img_path = vis_images[0]
                visualize_feature_maps(model, img_path, train_mean, train_std, file_name=imgs_file_name,
                                       folder=model_path, device=DEVICE, img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH)

                # Grad-CAM
                visualize_gradcam_UNET(model, vis_loader, file_name=imgs_file_name, folder=model_path, device=DEVICE)

                plt.close('all')

    #time end
    end_time = time.time()

    # print total training time in hours, minutes, seconds
    print("All epochs completed. Total training time: ", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))

    # save last state of the model
    # save model and sample predictions
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, model_dir=model_dir, model_name=f"{model_name}_200", parent_dir=parent_dir)

if __name__ == "__main__":
    # loop over main for the following parameters
    model_archs = ["UNet"]
    for model_arch in model_archs:
        main(model_arch)