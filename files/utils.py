import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def generate_model_name(architecture, loss_function, optimizer, dataset_names):
    model_name = f"{architecture}_{loss_function}_{optimizer}"
    for dataset_name in dataset_names:
        model_name += f"_{dataset_name}"
    return model_name

def save_checkpoint(state, filename="my_checkpoint.pth.tar", model_name=None, parent_dir=None):
    if model_name is not None:
        os.makedirs(os.path.join(parent_dir, "trained_models"), exist_ok=True)
        filepath = os.path.join(parent_dir, "trained_models")
        filename = os.path.join(parent_dir, "trained_models", f"{model_name}.pth.tar")

    torch.save(state, filename)
    print(f'Best model saved as {filename}')
    return filepath

def count_samples_in_loader(loader):
    total_samples = 0
    for images, masks in loader:
        total_samples += images.shape[0]
    return total_samples

def visualize_sample_images(train_loader, train_mean, train_std, batch_size, unorm):
    images, masks = next(iter(train_loader))
    n_samples = batch_size // 2

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))

    # Iterate over the images and masks and plot them side by side
    for i in range(n_samples):
        img = unorm(images[i].squeeze(0))
        img = np.transpose(img.numpy(), (1, 2, 0))
        mask = np.squeeze(masks[i].numpy(), axis=0)

        # Create an overlay of the mask on the image
        overlay = img.copy()
        overlay[mask == 1] = [1, 1, 1]  # Set the white pixels of the mask onto the image

        axs[i, 0].axis("off")
        axs[i, 0].imshow(img)
        axs[i, 1].axis("off")
        axs[i, 1].imshow(mask, cmap="gray")
        axs[i, 2].axis("off")
        axs[i, 2].imshow(overlay)
    plt.show()

def update_log_df(log_df, metric_dict, epoch, train_loss, scheduler):
    new_row = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0], "train_loss": train_loss, **metric_dict}
    log_df.loc[len(log_df)] = new_row
    return log_df

def save_predictions_as_imgs(loader, model, unnorm, model_name, folder="saved_images/", device="cuda",
                             testing=False, BATCH_SIZE=16):
    if testing == True:
        name_extension = "test"
    else:
        name_extension = "val"

    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    all_images = []
    for idx, (X, y) in enumerate(loader):
        num_images = X.size(0)
        X = X.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()

            X = X.cpu()
            y = y.cpu()
            preds = preds.cpu()

            for i in range(X.size(0)):
                X[i] = unnorm(X[i])

            X = (X - X.min()) / (X.max() - X.min())

            y = y.repeat(1, 3, 1, 1)
            preds = preds.repeat(1, 3, 1, 1)

            if num_images < BATCH_SIZE:
                pad_size = BATCH_SIZE - num_images
                zero_padding = torch.zeros((pad_size, 3, X.size(2), X.size(3)))
                X = torch.cat((X, zero_padding), dim=0)
                y = torch.cat((y, zero_padding), dim=0)
                preds = torch.cat((preds, zero_padding), dim=0)

            combined = torch.cat((X, y, preds), dim=3)
            all_images.append(combined)

            if idx == 2:
                break

    stacked_images = torch.cat(all_images, dim=2)

    path = os.path.join(folder, f"{model_name}_{name_extension}.png")
    torchvision.utils.save_image(stacked_images, path)

    model.train()

def check_non_binary_pixels(mask,transformation):
    tensor_mask = transforms.ToTensor()(mask)
    unique_values = torch.unique(tensor_mask)
    for value in unique_values:
        assert value == 0 or value == 1, f"After transformation: {transformation}; Mask contains non-binary value: {value}"

def get_mean_std(train_loader):
    """Function to calculate the mean and standard deviation of the training set.
    :param train_loader:
    :return: mean, std"""
    print('Retrieving mean and std for Normalization')
    train_mean = []
    train_std = []

    for batch_idx, (X, y) in enumerate(train_loader):
        numpy_image = X.numpy()

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))

        train_mean.append(batch_mean)
        train_std.append(batch_std)
    train_mean = torch.tensor(np.mean(train_mean, axis=0))
    train_std = torch.tensor(np.mean(train_std, axis=0))

    # convert mean and std to tuple
    print('Mean of Training Images:', train_mean)
    print('Std Dev of Training Images:', train_std)
    return train_mean, train_std

class UnNormalize(object):
    # link: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor