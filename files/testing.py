import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
from tabulate import tabulate
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
    get_loaders,
    save_predictions_as_imgs,
    calculate_binary_metrics,
    calculate_classification_metrics,
    generate_model_name
)


def load_model(model_name, model, parent_dir):
    model_path = os.path.join(parent_dir, "trained_models", f"{model_name}.pth.tar")

    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> Loaded checkpoint '{model_path}")
    return model_path

def main(model_name, dataset_fractions, train_mean, train_std, loss_fn, CROPPING=False, classification=False):
    # set seed
    RANDOM_SEED = 42

    # device agnostic
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # parameters
    if DEVICE == "cuda":
        NUM_WORKERS = 4
    else:
        NUM_WORKERS = 0
    IMAGE_HEIGHT = 416  # 400 originally
    IMAGE_WIDTH = 416  # 400 originally
    PIN_MEMORY = True
    BATCH_SIZE = 16

    # cwd
    cwd = os.getcwd()

    # parent directory
    parent_dir = os.path.dirname(cwd)

    image_dirs, mask_dirs, fractions = get_dirs_and_fractions(dataset_fractions, parent_dir)

    ############################
    # Train and validation splits
    ############################
    # Train and validation splits
    train_images, train_masks, val_images, val_masks, total_selected_images = create_train_val_splits(
        image_dirs,
        mask_dirs,
        fractions,
        val_size=1,
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
    assert (len(train_ds) + len(val_ds)) == (total_selected_images*2)
    print(f"Total number of images: {total_selected_images}, "
          f"of which {len(train_ds)} are training images and {len(val_ds)} are validation images.")
    del val_ds, train_ds

    ############################
    # Transforms
    ############################
    transformations = TransformationTypes(train_mean, train_std, IMAGE_HEIGHT, IMAGE_WIDTH, cropping=CROPPING)
    train_transforms = transforms.Lambda(transformations.apply_train_transforms)
    val_transforms = transforms.Lambda(transformations.apply_val_transforms)

    # Get loaders
    _, val_loader = get_loaders(
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

    # retrieve first expression up until first "_" appearance
    model_arch = model_name.split("_")[0]

    if model_arch == "UNet":
        model = UNet(in_channels=3, out_channels=1)
    else:
        model = create_segformer(model_arch, channels=3, num_classes=1).to(DEVICE)
    model.to(DEVICE)

    # Load the pretrained model weights and parameters by model name
    model_path = load_model(model_name, model, parent_dir)
    model_folder = os.path.dirname(model_path)

    # model summary
    summary(model, input_size=(BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    if classification == True:
        # Calculate classification metrics
        classification_metrics = calculate_classification_metrics(val_loader, model, DEVICE)
        acc, f1, precision, recall, cm = classification_metrics
        tn, fp, fn, tp = cm.ravel()
        print(
            f"Classification Metrics: Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        # Print the confusion matrix with row and column labels
        confusion_matrix_table = [
            ["", "Actual"],
            ["Predicted", "1", "0"],
            ["1", f"TP: {tp}", f"FP: {fp}"],
            ["0", f"FN: {fn}", f"TN: {tn}"],
        ]
        print(tabulate(confusion_matrix_table, tablefmt="fancy_grid"))
    else:
        # Calculate segmentation metrics
        avg_metrics = calculate_binary_metrics(val_loader, model, loss_fn, device=DEVICE)
        pixel_acc, iou, precision, specificity, recall, f1_score, bg_acc, val_loss = avg_metrics
        print(
            f"Test Metrics: F1-Score:{f1_score:.4f} | Recall:{recall:.4f} | Precision:{precision:.4f} | Pixel-Acc: {pixel_acc:.4f} | Loss: {val_loss:.4f}")

    # unnormalize
    unorm = UnNormalize(mean=tuple(train_mean.numpy()), std=(tuple(train_std.numpy())))


    save_predictions_as_imgs(
        val_loader, model, unnorm=unorm, model_name=model_name, folder=model_folder,
        device=DEVICE, testing=True, BATCH_SIZE=BATCH_SIZE)

if __name__ == '__main__':

    # specify model name
    model_name = "B0_DiceBCELoss_AdamW_France_google_Munich_Denmark_Heerlen_2018_HR_output_ZL_2018_HR_output"

    # specify test dataset(s)
    dataset_fractions = [
        # [dataset_name, fraction_of_positivies, fraction_of_negatives]
        # ['France_google', 0.005, 0.0],
        # ['Munich', 0.0, 0.0],
        # ['Denmark', 0.0, 0.001],
        ['Heerlen_2018_HR_output', 0, 0.001],
        # ['ZL_2018_HR_output', 0, 1]
    ]

    # Specify Cropping or No Cropping
    CROP = True
    # Specify Classification or Segmentation
    CLASSIFICATION = False

    # specify loss
    loss_fn = DiceBCELoss()

    # specify train_mean, train_std -> in format tensor([0.2929, 0.2955, 0.2725]), tensor([0.2268, 0.2192, 0.2098])
    train_mean = torch.tensor([0.3542, 0.3581, 0.3108])
    train_std = torch.tensor([0.2087, 0.1924, 0.1857])

    # run main function
    main(model_name, dataset_fractions, train_mean, train_std, loss_fn, CROP, CLASSIFICATION)