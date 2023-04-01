import torch
import torchvision
from dataset import FranceSegmentationDataset
from torch.utils.data import DataLoader
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    image_dir,
    mask_dir,
    train_images,
    train_masks,
    val_images,
    val_masks,
    batch_size,
    train_transforms,
    val_transforms,
    num_workers=0,
    pin_memory=True,
):
    train_ds = FranceSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        images=train_images,
        masks=train_masks,
        transform=train_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = FranceSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    class_correct = [0, 0]  # [background_correct, foreground_correct]
    class_pixels = [0, 0]  # [background_pixels, foreground_pixels]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum().item()) / (
                (preds + y).sum().item() + 1e-8
            )  # Convert to Python int

            # Overall accuracy calculation
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            # Class-wise accuracy calculation
            for cls in range(2):
                class_correct[cls] += ((preds == y) * (y == cls).float()).sum().item()
                class_pixels[cls] += (y == cls).sum().item()

    print(f"Dice score: {dice_score/len(loader):.3f}")
    print(f"Overall Accuracy: {num_correct/num_pixels*100:.2f}")
    print(f"Background class accuracy: {class_correct[0]/class_pixels[0]*100:.2f}")
    print(f"Foreground class accuracy: {class_correct[1]/class_pixels[1]*100:.2f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    # create a folder if not exists, cwd + folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    # set model to eval mode
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Move x, y, and preds back to the CPU
        x = x.cpu()
        y = y.cpu()
        preds = preds.cpu()

        # Normalize the input image back to the range [0, 1]
        x = (x - x.min()) / (x.max() - x.min())

        # Repeat the single channel of y and preds 3 times to match the number of channels in x
        y = y.repeat(1, 3, 1, 1)
        preds = preds.repeat(1, 3, 1, 1)

        # Concatenate the image, ground truth mask, and prediction along the width dimension (dim=3)
        combined = torch.cat((x, y, preds), dim=3)

        # Save the combined image
        torchvision.utils.save_image(combined, f"{folder}/combined_{idx}.png")

        # Break after the first batch
        if idx == 0:
            break

    # Set the model back to train mode
    model.train()