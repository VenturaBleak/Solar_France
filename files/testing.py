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
from dataset import (FranceSegmentationDataset, TransformationTypes, get_loaders,
                     create_train_val_splits,get_dirs_and_fractions, filter_positive_images,
                     UnNormalize, get_mean_std)
from train import train_fn
from image_size_check import check_dimensions
from utils import (
    save_checkpoint,
    BinaryMetrics,
    generate_model_name,
    visualize_sample_images,
    save_predictions_as_imgs,
    count_samples_in_loader,
    calculate_classification_metrics
)
from solar_snippet_v2 import ImageProcessor
from tabulate import tabulate



def load_model(model_name, model, parent_dir):
    model_path = os.path.join(parent_dir, "trained_models", f"{model_name}.pth.tar")

    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> Loaded checkpoint '{model_path}")
    return model_path

def main(model_name, dataset_fractions, train_mean, train_std, loss_fn, crop=False, classification=False,
         probability_threshold=0.5, pixel_threshold=0.001, grid_search=False, min_probability_threshold=0.0000,
         max_probability_threshold=0.01, incremental_probability_step=0.0002, min_pixel_threshold=0.0000,
         max_pixel_threshold=0.01, incremental_pixel_step=0.0002):
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


    ############################
    # Transforms
    ############################
    transformations = TransformationTypes(train_mean, train_std, IMAGE_HEIGHT, IMAGE_WIDTH, cropping=crop)
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

    print(f"Total number of testing images: {total_selected_images}")
    del val_ds, train_ds

    # Evaluate: Classification OR Segmentation
    if classification:
        if grid_search:
            best_f1 = 0
            best_probability_threshold = min_probability_threshold
            best_pixel_threshold = min_pixel_threshold
            # Loop over the different probability thresholds
            num_probability_steps = int(
                (max_probability_threshold - min_probability_threshold) / incremental_probability_step) + 1
            for i in range(num_probability_steps):
                current_probability_threshold = min_probability_threshold + i * incremental_probability_step

                # Loop over the different pixel thresholds
                num_pixel_steps = int((max_pixel_threshold - min_pixel_threshold) / incremental_pixel_step) + 1
                for j in range(num_pixel_steps):
                    current_pixel_threshold = min_pixel_threshold + j * incremental_pixel_step

                    # Calculate classification metrics
                    classification_metrics = calculate_classification_metrics(val_loader, model,
                                                                              current_probability_threshold,
                                                                              current_pixel_threshold, DEVICE)
                    acc, f1, precision, recall, cm = classification_metrics

                    # Check if the current thresholds have a better F1-score
                    if f1 > best_f1:
                        best_f1 = f1
                        best_probability_threshold = current_probability_threshold
                        best_pixel_threshold = current_pixel_threshold
                        print(
                            f"Probability Threshold: {current_probability_threshold:.4f}, Pixel Threshold: {current_pixel_threshold:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

            print(f"\n Grid search complete! \n"
                  f"Best probability threshold: {best_probability_threshold:.4f}, "
                  f"Best pixel threshold: {best_pixel_threshold:.4f}, "
                  f"Best F1-score: {best_f1:.4f}, "
                  f"Best Accuracy: {acc:.4f}")
        else:
            # Calculate classification metrics
            classification_metrics = calculate_classification_metrics(val_loader, model, probability_threshold,
                                                                      pixel_threshold, DEVICE)
            acc, f1, precision, recall, cm = classification_metrics
            print(
                f"Classification Metrics: Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Probability Threshold: {probability_threshold:.4f} | Pixel Threshold: {pixel_threshold:.4f}")

            # Extract the true positives, false positives, true negatives and false negatives from the confusion matrix
            tp, fp, tn, fn = cm.ravel()

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
        binary_metrics = BinaryMetrics()
        metric_dict = binary_metrics.calculate_binary_metrics(val_loader, model, loss_fn, device=DEVICE)
        print(
            f"Test.Metrics: Loss: {metric_dict['val_loss']:.4f} | Balanced-Acc:{metric_dict['balanced_acc']:.3f} | "
            f"F1-Score:{metric_dict['f1_score']:.3f} | Precision:{metric_dict['precision']:.3f} | "
            f"Recall:{metric_dict['recall']:.3f}"
        )

    # unnormalize
    unorm = UnNormalize(mean=tuple(train_mean.numpy()), std=(tuple(train_std.numpy())))

    save_predictions_as_imgs(
        val_loader, model, unnorm=unorm, model_name=model_name, folder=model_folder,
        device=DEVICE, testing=True, BATCH_SIZE=BATCH_SIZE)

if __name__ == '__main__':

    # specify model name
    model_name = "B0_TverskyLoss_AdamW_France_google_Munich_Denmark_Heerlen_2018_HR_output_ZL_2018_HR_output"

    # specify test dataset(s)
    dataset_fractions = [
        # [dataset_name, fraction_of_positivies, fraction_of_negatives]
        # ['France_google', 0.002, 0.0],
        # ['Munich', 0.0, 0.0],
        # ['Denmark', 0.0, 0.001],
        ['Heerlen_2018_HR_output', 0.005, 0.002],
        # ['ZL_2018_HR_output', 0, 1]
    ]

    # Specify Cropping or No Cropping
    CROP = True
    # Specify Classification or Segmentation
    CLASSIFICATION = True
    PROBABILITY_THRESHOLD = 0.5
    PIXEL_THRESHOLD = 0.001

    # Specify Grid Search
    GRID_SEARCH = True
    MIN_PROBABILITY_THRESHOLD = 0.2
    MAX_PROBABILITY_THRESHOLD = 0.8
    INCREMENTAL_PROBABILITY_STEP = 0.1
    MIN_PIXEL_THRESHOLD = 0.0001
    MAX_PIXEL_THRESHOLD = 0.1
    INCREMENTAL_PIXEL_STEP = 0.005

    # specify loss
    loss_fn = IoULoss()

    # specify train_mean, train_std -> in format tensor([0.2929, 0.2955, 0.2725]), tensor([0.2268, 0.2192, 0.2098])
    train_mean = torch.tensor([0.3542, 0.3581, 0.3108])
    train_std = torch.tensor([0.2087, 0.1924, 0.1857])

    # run main function
    main(model_name, dataset_fractions, train_mean, train_std, loss_fn, CROP, CLASSIFICATION, PROBABILITY_THRESHOLD,
         PIXEL_THRESHOLD, GRID_SEARCH, MIN_PROBABILITY_THRESHOLD, MAX_PROBABILITY_THRESHOLD,
         INCREMENTAL_PROBABILITY_STEP, MIN_PIXEL_THRESHOLD, MAX_PIXEL_THRESHOLD, INCREMENTAL_PIXEL_STEP)