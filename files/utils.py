import os
import glob
import re
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageSequence, ImageDraw, ImageFont


def generate_model_name(architecture, loss_function, optimizer, dataset_names):
    model_name = f"{architecture}_{loss_function}_{optimizer}"
    for dataset_name in dataset_names:
        model_name += f"_{dataset_name}"
    return model_name

def save_checkpoint(state, filename="my_checkpoint.pth.tar", model_dir =None, model_name=None,  parent_dir=None):
    # raise error if either model_dir or parent_dir is None
    try:
        assert model_dir is not None or parent_dir is not None
    except AssertionError:
        raise AssertionError("Either model_dir or parent_dir must be specified")

    # if model_dir is not None, save model in model_dir
    if model_name is not None:
        os.makedirs(os.path.join(parent_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, "models", model_dir), exist_ok=True)
        filepath = os.path.join(parent_dir, "models", model_dir)
        filename = os.path.join(parent_dir, "models", model_dir, f"{model_name}.pth.tar")

    torch.save(state, filename)
    print(f'Best model saved as {filename}')
    return filepath

def load_model(model_dir, model_name, model, parent_dir):
    model_path = os.path.join(parent_dir, "models", model_dir, f"{model_name}.pth.tar")

    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> Loaded checkpoint '{model_path}")
    return model_path

def count_samples_in_loader(loader):
    total_samples = 0
    for images, masks in loader:
        total_samples += images.shape[0]
    return total_samples

def visualize_sample_images(train_loader, train_mean, train_std, batch_size, unorm):
    images, masks = next(iter(train_loader))
    n_samples = batch_size // 2
    image_size = images.shape[2]  # assuming images are square

    # Adjust the figure size to match the aspect ratio of the images.
    fig, axs = plt.subplots(n_samples, 2, figsize=(10, n_samples * 5))

    for i in range(n_samples):
        img = unorm(images[i].squeeze(0))
        img = np.transpose(img.numpy(), (1, 2, 0))
        mask = np.squeeze(masks[i].numpy(), axis=0)

        overlay = img.copy()
        overlay[mask == 1] = [1, 0.5, 0]

        overlay_img = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay_img[mask == 1] = matplotlib.colors.to_rgba('purple', alpha=0.35)
        overlay_img[mask == 0] = matplotlib.colors.to_rgba('none', alpha=0)

        axs[i, 0].axis("off")
        axs[i, 0].imshow(img, aspect="auto")
        axs[i, 1].axis("off")
        axs[i, 1].imshow(img, aspect="auto")
        axs[i, 1].imshow(overlay_img, aspect="auto")

    # Remove padding and white background
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def update_log_df(log_df, metric_dict, epoch, train_loss, scheduler):
    new_row = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0], "train_loss": train_loss, **metric_dict}
    log_df.loc[len(log_df)] = new_row
    return log_df

def overlay_on_image(img, mask, color, alpha=0.35):
    # convert tensors to PIL Images
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    overlay = Image.fromarray((img * 255).astype(np.uint8))

    # create overlay
    overlay_np = np.array(overlay)
    overlay_np[mask == 1] = np.array(color) * 255  # Multiply by 255 as PIL uses 0-255 range
    overlay = Image.fromarray(overlay_np)

    # blend images
    blended = Image.blend(img_pil, overlay, alpha)

    # convert back to tensor
    blended = np.transpose(np.array(blended), (2, 0, 1))
    return torch.from_numpy(blended).float() / 255.  # Divide by 255 to convert back to 0-1 range

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

            y_overlays = []
            preds_overlays = []

            for i in range(X.size(0)):
                X[i] = unnorm(X[i])
                img = np.transpose(X[i].numpy(), (1, 2, 0))

                y_overlays.append(overlay_on_image(img, y[i, 0], [0, 1, 0]))  # Overlay for y in teal
                preds_overlays.append(overlay_on_image(img, preds[i, 0], [1, 0, 1]))  # Overlay for preds in purple

            X = (X - X.min()) / (X.max() - X.min())
            y_overlays = torch.stack(y_overlays)
            preds_overlays = torch.stack(preds_overlays)

            if num_images < BATCH_SIZE:
                pad_size = BATCH_SIZE - num_images
                zero_padding = torch.zeros((pad_size, 3, X.size(2), X.size(3)))
                X = torch.cat((X, zero_padding), dim=0)
                y_overlays = torch.cat((y_overlays, zero_padding), dim=0)
                preds_overlays = torch.cat((preds_overlays, zero_padding), dim=0)

            combined = torch.cat((X, y_overlays, preds_overlays), dim=3)
            all_images.append(combined)

            if idx == 2:
                break

        stacked_images = torch.cat(all_images, dim=2)

        path = os.path.join(folder, f"{model_name}_{name_extension}.png")
        torchvision.utils.save_image(stacked_images, path)

        model.train()

def create_gif_from_images(image_folder, image_name_pattern, output_gif_name, image_index, img_height, img_width,
                           font_path, num_epochs):
    """
    Create a GIF from a subset of images in a folder.

    Args:
        image_folder (str): Path to the folder containing the images.
        image_name_pattern (str): Pattern to match the image names.
        output_gif_name (str): Name of the output GIF file.
        image_index (int): Index of the image-mask-prediction combination to be included in the GIF.
        img_height (int): Height of the individual image.
        img_width (int): Width of the individual image.
        font_path (str): Path to the .ttf file for the desired font.

    Returns:
        None
    """
    # Get list of all files in the image_folder
    all_files = os.listdir(image_folder)

    # Initialize a list to hold the matching image files
    image_files = {}

    # Loop over all files and match the filenames against the pattern
    for filename in all_files:
        match = re.match(image_name_pattern, filename)
        if match:
            epoch_num = int(match.group(1))  # get epoch number from match group
            image_files[epoch_num] = os.path.join(image_folder, filename)

    # Initialize a list to hold the selected images
    selected_images = []

    # Calculate the x coordinates of the slice
    x_start = image_index * (img_width + 1)
    x_end = x_start + img_width

    # Loop over the image files sorted by epoch
    for epoch in sorted(image_files.keys()):  # sort to maintain order
        # Open the image file
        img = Image.open(image_files[epoch])

        # Cut out the selected image-mask-prediction combination
        selected_img = img.crop((x_start, 0, x_end, img_height))

        # Create a draw object and specify the font size and color
        draw = ImageDraw.Draw(selected_img)
        font = ImageFont.truetype(font_path, 18)  # You may need to adjust the font size

        # Add text to the bottom right corner
        text = f"Epoch {epoch}"
        textwidth, textheight = draw.textsize(text, font)
        width, height = selected_img.size
        x = width - textwidth - 25  # 25 pixels from the right
        y = height - textheight - 10  # 10 pixels from the bottom

        # Draw a semi-transparent rectangle behind the text
        rectangle_left = x - 5
        rectangle_top = y - 5
        rectangle_right = x + textwidth + 5
        rectangle_bottom = y + textheight + 5
        draw.rectangle([rectangle_left, rectangle_top, rectangle_right, rectangle_bottom], fill=(0, 0, 0))

        # Draw the text
        draw.text((x, y), text, font=font, fill='white')

        # Append to the list
        selected_images.append(selected_img)

    # Create the GIF
    if selected_images:  # only if the list is not empty
        selected_images[0].save(os.path.join(image_folder, output_gif_name), save_all=True,
                                append_images=selected_images[1:], optimize=False, duration=num_epochs * 10, loop=0,
                                dither=Image.FLOYDSTEINBERG)
        print(f"Created GIF '{output_gif_name}' with {len(selected_images)} images.")
    else:
        print("No images found to create a GIF.")

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