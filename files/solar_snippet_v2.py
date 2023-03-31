import os  # Import the os module for handling file paths
import numpy as np  # Import numpy for array manipulation
import scipy.ndimage as ndi  # Import the ndimage module from SciPy for image processing
from PIL import Image  # Import the Image module from PIL for handling images

def find_largest_white_patch(mask):
    """Function to find the largest white patch in a binary mask
    :param mask: Binary mask, where white pixels (solar PVs) are 1 and black pixels are 0
    :return: Center (x,y coordinates wihtin mask) and bounding box of the largest white patch
    """
    mask_labeled, num_features = ndi.label(mask)  # Label connected components in the mask

    if num_features == 0:  # If there are no connected components, return None
        return None, None

    patch_sizes = np.bincount(mask_labeled.flat)[1:]  # Count the size of each connected component
    largest_patch = np.argmax(patch_sizes) + 1  # Find the label of the largest connected component
    largest_patch_pixels = np.nonzero(mask_labeled == largest_patch)  # Get the pixel coordinates of the largest connected component

    center = np.mean(np.column_stack(largest_patch_pixels), axis=0).astype(np.int32)  # Calculate the center of the largest connected component
    bounding_box = (np.min(largest_patch_pixels[1]), np.min(largest_patch_pixels[0]), np.max(largest_patch_pixels[1]), np.max(largest_patch_pixels[0]))  # Calculate the bounding box of the largest connected component

    return tuple(center), bounding_box  # Return the center and bounding box

def crop_solar_panel(image, mask, size=50):
    """Function to crop a solar panel from the image and its mask
    :param image: Image to crop
    :param mask: Mask to crop
    :param size: Size of the crop
    :return: Cropped image and mask
    """
    mask_np = mask.copy()  # Create a copy of the mask

    if np.sum(mask_np) == 0:  # If there are no white pixels in the mask, raise an exception
        raise ValueError("No white pixels in solar panel mask.")

    labeled_mask, num_features = ndi.label(mask_np)  # Label connected components in the mask
    largest_cc = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1  # Find the label of the largest connected component
    largest_cc_mask = (labeled_mask == largest_cc).astype(np.uint8) * 255  # Create a binary mask of the largest connected component

    white_pixels = np.nonzero(largest_cc_mask)  # Get the pixel coordinates of white pixels in the largest connected component mask
    bounding_box = (np.min(white_pixels[1]), np.min(white_pixels[0]), np.max(white_pixels[1]), np.max(white_pixels[0]))  # Calculate the bounding box of the largest connected component
    cropped_image = image.crop(bounding_box)  # Crop the image using the bounding box
    cropped_mask = Image.fromarray(largest_cc_mask[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]])  # Crop the mask using the bounding box

    return cropped_image, cropped_mask  # Return the cropped image and mask

def paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, location, bounding_box):
    """Function to paste a solar panel onto a target image and its mask"""
    x, y = location  # Unpack the x and y coordinates of the target location
    x_min, y_min, x_max, y_max = bounding_box  # Unpack the bounding box of the target area

    paste_x = max(min(x - cropped_image.size[0] // 2, x_max - cropped_image.size[0]), x_min)  # Calculate the x-coordinate of the paste location
    paste_y = max(min(y - cropped_image.size[1] // 2, y_max - cropped_image.size[1]), y_min)  # Calculate the y-coordinate of the paste location

    target_image.paste(cropped_image, (paste_x, paste_y), mask=cropped_mask)  # Paste the cropped image onto the target image using the mask
    target_mask.paste(cropped_mask, (paste_x, paste_y), mask=cropped_mask)  # Paste the cropped mask onto the target mask using the mask

def modify_images(source_image, source_mask, target_image, target_mask):
    """
    Function to modify the target images and masks by pasting a solar panel from the source images and masks
    :param source_image: Source image -> this is the cropped image of the solar panel
    :param source_mask: Source mask -> this is the cropped mask of the solar panel
    :param target_image: Target image -> this is the image that will be modified
    :param target_mask: Target mask -> this is the mask that will be modified
    :return: Modified target image and mask, now with a solar panel pasted onto it
    """
    if np.sum(target_mask) == 0:  # If there are no white pixels in the target mask, raise an exception
        raise ValueError("No white pixels in building mask.")

    source_mask_np = np.array(source_mask, dtype=np.uint8)  # Convert the source mask to a numpy array
    source_mask_np[source_mask_np > 0] = 255  # Set all non-zero values to 255

    target_mask_np = np.array(target_mask, dtype=np.uint8)  # Convert the target mask to a numpy array
    target_mask_np[target_mask_np > 0] = 255  # Set all non-zero values to 255

    # Find the largest white patch in the target mask
    largest_patch_location, bounding_box = find_largest_white_patch(target_mask_np)  # Find the largest white patch in the target mask

    # Clear the original mask, i.e. fill it with 0s, that is, remove the building segmentations
    target_mask_np.fill(0)

    # Convert the array back to an image
    target_mask = Image.fromarray(target_mask_np.astype(np.uint8))

    # Crop the solar panel from the source image and its mask
    cropped_image, cropped_mask = crop_solar_panel(source_image, source_mask_np)

    # Paste the solar panel onto the target image and its mask
    paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, largest_patch_location, bounding_box)

    # Return the modified target image and its mask
    return target_image, target_mask


if __name__ == "__main__":
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)

    image_dir = os.path.join(parent_dir, 'data', 'trial', 'images')
    mask_dir = os.path.join(parent_dir, 'data', 'trial', 'masks')

    source_image = Image.open(os.path.join(image_dir, "ABEKP6A34YFLOO.png")).convert("RGB")
    source_mask = Image.open(os.path.join(mask_dir, "ABEKP6A34YFLOO.png")).convert("L")
    target_image = Image.open(os.path.join(image_dir, "AAYQM71ADIEGMN.png")).convert("RGB")
    target_mask = Image.open(os.path.join(mask_dir, "AAYQM71ADIEGMN.png")).convert("L")

    try:
        # most decisive line, later on we will use this function to modify the images within the data loader in the training loop
        modified_target_image, modified_target_mask = modify_images(source_image, source_mask, target_image, target_mask)

        modified_target_image.save(os.path.join(image_dir, "modified_target_image.png"))
        modified_target_mask.save(os.path.join(mask_dir, "modified_target_mask.png"))

        modified_target_image.show()
        modified_target_mask.show()

    except NoWhitePixelsException as e:
        print(e)