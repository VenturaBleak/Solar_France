from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def test_mask_overlay(image_file, mask_file):
    # Load the image and mask with PIL
    image = Image.open(image_file).convert('RGBA')  # convert to RGBA
    mask = Image.open(mask_file).convert('L')  # convert to grayscale

    # Convert the PIL Images to numpy arrays for manipulation
    image = np.array(image)
    mask = np.array(mask)

    # Create a semi-transparent color for the mask
    overlay_color = [255, 0, 255, 128]  # RGBA: last value is alpha for transparency

    # Create the mask image in RGBA format
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[mask == 255] = overlay_color

    # Combine the image and the mask
    overlay = np.where(mask_rgba != 0, mask_rgba, image)

    # Convert the overlay back to PIL Image
    overlay_pil = Image.fromarray(overlay)

    # Resize the image for zooming (150%)
    overlay_zoom = overlay_pil.resize((int(overlay_pil.size[0]*1.5), int(overlay_pil.size[1]*1.5)), Image.LANCZOS)
    overlay = np.array(overlay_zoom)

    # Display the overlay
    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(overlay)
    ax.axis('off')  # Turn off axis
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()

if __name__ == "__main__":
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # Heerlen or ZL
    REGION = "Heerlen_2018_HR_output"

    # Define output folders
    mask_folder = os.path.join(parent_dir, "data_NL", REGION, "masks_positive")  # Replace "directory_to_specify" with the actual directory
    image_folder = os.path.join(parent_dir, "data_NL", REGION, "images_positive")  # Replace "directory_to_specify" with the actual directory

    # The number of images to process
    x = 5  # for example

    # Get all the image files in the directory
    image_files = os.listdir(image_folder)

    # Ensure we have corresponding mask files for each image
    mask_files = [img_file for img_file in image_files if os.path.exists(os.path.join(mask_folder, img_file))]

    # Loop over the first x image and mask files
    for i in range(min(x, len(mask_files))):
        # Get the full path of the image and mask files
        image_file = os.path.join(image_folder, mask_files[i])
        mask_file = os.path.join(mask_folder, mask_files[i])

        # Call the test
        test_mask_overlay(image_file, mask_file)