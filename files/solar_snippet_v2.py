import os  # Import the os module for handling file paths
import numpy as np  # Import numpy for array manipulation
import scipy.ndimage as ndi  # Import the ndimage module from SciPy for image processing
from PIL import Image  # Import the Image module from PIL for handling images
import math
from skimage.measure import regionprops_table
import time
import random

def find_largest_white_patch(mask):
    """Function to find the largest white patch in a binary mask
    :param mask: Binary mask, where white pixels (solar PVs) are 1 and black pixels are 0
    :return: Center (x,y coordinates wihtin mask) and bounding box of the largest white patch
    """
    mask_labeled, num_features = ndi.label(mask)  # Label connected components in the mask

    patch_sizes = np.bincount(mask_labeled.flat)[1:]  # Count the size of each connected component
    largest_patch = np.argmax(patch_sizes) + 1  # Find the label of the largest connected component
    largest_patch_pixels = np.nonzero(mask_labeled == largest_patch)  # Get the pixel coordinates of the largest connected component

    center = np.mean(np.column_stack(largest_patch_pixels), axis=0).astype(np.int32)  # Calculate the center of the largest connected component
    bounding_box = (np.min(largest_patch_pixels[1]), np.min(largest_patch_pixels[0]), np.max(largest_patch_pixels[1]), np.max(largest_patch_pixels[0]))  # Calculate the bounding box of the largest connected component

    return tuple(center), bounding_box  # Return the center and bounding box

def find_random_white_patch(mask, min_pixel_count=100):
    """
    Function to find a random white patch from the list sorted by their size in a binary mask,
    excluding patches with less than min_pixel_count pixels.
    :param mask: Binary mask, where white pixels (solar PVs) are 1 and black pixels are 0
    :param min_pixel_count: Minimum number of pixels in a patch to be considered for random selection (default: 100)
    :return: Center (x,y coordinates within mask) and bounding box of the randomly selected white patch
    """
    mask_labeled, num_features = ndi.label(mask)  # Label connected components in the mask

    patch_sizes = np.bincount(mask_labeled.flat)[1:]  # Count the size of each connected component
    valid_patches_indices = np.where(patch_sizes >= min_pixel_count)[0]  # Filter patches based on the minimum pixel count

    # If there are valid patches, randomly select a patch from the valid patches
    try:
        total_white_pixels = np.sum(patch_sizes[valid_patches_indices])  # Calculate the total number of white pixels in the valid patches
        probabilities = patch_sizes[valid_patches_indices] / total_white_pixels  # Calculate the proportional probabilities

        # Randomly choose a patch from the valid_patches_indices list using proportional probabilities
        selected_patch = np.random.choice(valid_patches_indices, p=probabilities) + 1

        # Alternative: randomly choose a patch from the valid patches, by simple random sampling, wihtout probabilities
        # selected_patch = np.random.choice(valid_patches_indices) + 1

        selected_patch_pixels = np.nonzero(mask_labeled == selected_patch)  # Get the pixel coordinates of the selected patch

        center = np.mean(np.column_stack(selected_patch_pixels), axis=0).astype(np.int32)  # Calculate the center of the selected patch
        bounding_box = (np.min(selected_patch_pixels[1]), np.min(selected_patch_pixels[0]), np.max(selected_patch_pixels[1]), np.max(selected_patch_pixels[0]))  # Calculate the bounding box of the selected patch

    # If there are no valid patches, randomly select a center and bounding box
    except:
        height, width = mask.shape
        center = (random.randint(0, width - 1), random.randint(0, height - 1))
        bounding_box = (center[0] - 30, center[1] - 30, center[0] + 30, center[1] + 30)

    return tuple(center), bounding_box  # Return the center and bounding box

def crop_solar_panel(image, mask):
    """Function to crop a solar panel from the image and its mask
    :param image: Image to crop
    :param mask: Mask to crop
    :return: Cropped image and mask
    """
    mask_np = mask.copy()  # Create a copy of the mask

    # Convert the mask to a numpy array
    try:
        labeled_mask, num_features = ndi.label(mask_np)  # Label connected components in the mask
        largest_cc = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1  # Find the label of the largest connected component
        largest_cc_mask = (labeled_mask == largest_cc).astype(np.uint8) * 255  # Create a binary mask of the largest connected component
    # handle edge case where there are no white pixels in the mask
    except:
        # Return empty image and mask instead of raising an error
        empty_image = Image.new("RGB", (1, 1))
        empty_mask = Image.new("1", (1, 1))
        return empty_image, empty_mask

    white_pixels = np.nonzero(largest_cc_mask)  # Get the pixel coordinates of white pixels in the largest connected component mask
    bounding_box = (np.min(white_pixels[1]), np.min(white_pixels[0]), np.max(white_pixels[1]), np.max(white_pixels[0]))  # Calculate the bounding box of the largest connected component
    cropped_image = image.crop(bounding_box)  # Crop the image using the bounding box
    cropped_mask = Image.fromarray(largest_cc_mask[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]])  # Crop the mask using the bounding box

    # show both the cropped image and mask
    # cropped_image.show()
    # cropped_mask.show()

    return cropped_image, cropped_mask  # Return the cropped image and mask

def find_angle(mask):
    #show mask
    # mask.show()
    mask_array = np.array(mask)
    binary_mask = (mask_array == 255).astype(int)
    props = regionprops_table(binary_mask, properties=['orientation'])
    print(props)
    angle = props['orientation'][0]
    return angle

def paste_solar_panel(target_image, target_mask, target_mask_np, cropped_image, cropped_mask, location, bounding_box):
    """This function takes a target image and its corresponding mask, a cropped image of a solar panel with its
    corresponding mask, the location where the solar panel should be pasted, and the bounding box of the target area.
    It then aligns the cropped image with the target image by calculating the angle difference between them and
    rotating the cropped image and mask accordingly. Finally, it pastes the cropped image and mask onto the target
    image and mask, and returns the modified target mask.
    :param target_image: Target image
    :param target_mask: Target mask
    :param target_mask_np: Target mask as a numpy array
    :param cropped_image: Cropped image
    :param cropped_mask: Cropped mask
    :param location: Location where the cropped image should be pasted
    :param bounding_box: Bounding box of the target area
    :return: Modified target mask
    """

    # Get the bounding box coordinates
    x_min, y_min, x_max, y_max = bounding_box

    # Calculate the center of the bounding box
    center_x = (x_max + x_min) // 2
    center_y = (y_max + y_min) // 2

    # Calculate the position where the cropped image should be pasted
    paste_x = center_x - cropped_image.size[0] // 2
    paste_y = center_y - cropped_image.size[1] // 2

    # Introduce randomness to the paste position
    paste_x += random.randint(-25, 25)
    paste_y += random.randint(-25, 25)

    # Ensure the paste position is within the bounding box
    paste_x = max(x_min, min(paste_x, x_max - cropped_image.size[0]))
    paste_y = max(y_min, min(paste_y, y_max - cropped_image.size[1]))

    # Calculate the angle difference between target_mask and cropped_mask
    target_angle = find_angle(target_mask)
    cropped_angle = find_angle(cropped_mask)
    rotation = np.degrees(target_angle - cropped_angle)

    # Introduce randomness to the paste angle
    rotation += random.randint(-15, 15)

    # Rotate the cropped image and mask to align with the target mask
    cropped_image = cropped_image.rotate(rotation, resample=Image.BICUBIC, expand=True)
    cropped_mask = cropped_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

    # Clear the target mask, i.e. fill it with 0s, that is, remove the building segmentations
    target_mask_np.fill(0)

    # Convert the array back to an image, since we need to paste the cropped mask onto the cleared target mask
    target_mask = Image.fromarray(target_mask_np.astype(np.uint8))

    # Paste the cropped image and mask onto the target image and mask
    target_image.paste(cropped_image, (paste_x, paste_y), mask=cropped_mask)
    target_mask.paste(cropped_mask, (paste_x, paste_y), mask=cropped_mask)

    return target_mask


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

    # random sample from white patches with at minimum min_pixel_count pixels of white pixels on the target mask
    # experimented with the min_pixel_count parameter and 1500 seems to work well, given the 400x400 size of the target images
    MIN_PIXEL_COUNT = 1500
    largest_patch_location, bounding_box = find_random_white_patch(target_mask_np, min_pixel_count=MIN_PIXEL_COUNT)


    # alternative: always use the largest patch
    # largest_patch_location, bounding_box = find_largest_white_patch(target_mask_np)

    # Replace white values outside the bounding box with black values
    x_min, y_min, x_max, y_max = bounding_box
    target_mask_np[0:y_min, :] = 0
    target_mask_np[y_max:, :] = 0
    target_mask_np[:, 0:x_min] = 0
    target_mask_np[:, x_max:] = 0

    target_mask = Image.fromarray(target_mask_np)

    # Crop the solar panel from the source image and its mask
    cropped_image, cropped_mask = crop_solar_panel(source_image, source_mask_np)

    # Paste the solar panel onto the target image and its mask
    target_mask = paste_solar_panel(target_image, target_mask, target_mask_np, cropped_image, cropped_mask, largest_patch_location, bounding_box)

    # Return the modified target image and its mask
    return target_image, target_mask

if __name__ == "__main__":
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)

    solar_image_dir = os.path.join(parent_dir, 'data', 'France_google', 'images_positive')
    solar_mask_dir = os.path.join(parent_dir, 'data', 'France_google', 'masks_positive')

    building_image_dir = os.path.join(parent_dir, 'data', 'Munich_rooftops_noPV', 'images')
    building_mask_dir = os.path.join(parent_dir, 'data', 'Munich_rooftops_noPV', 'building_masks')

    destination_dir = os.path.join(parent_dir, 'data', 'trial', 'modified')

    # Get the first 10 solar image and mask filenames
    solar_image_files = sorted(os.listdir(solar_image_dir))[90:110]

    # Get the first 10 building image and mask filenames
    building_image_files = sorted(os.listdir(building_image_dir))[12:25]

    # time this process
    start = time.time()

    # for building_image_file in building_image_files:
    #     for solar_image_file in solar_image_files:



    # Iterate over the first 10 elements in the solar_image_dir and building_image_dir folders
    for solar_image_file, building_image_file, in zip(
            solar_image_files, building_image_files
    ):
        for i in range(1):


            source_image = Image.open(os.path.join(solar_image_dir, solar_image_file)).convert("RGB")
            source_mask = Image.open(os.path.join(solar_mask_dir, solar_image_file)).convert("L")
            target_image = Image.open(os.path.join(building_image_dir, building_image_file)).convert("RGB")
            target_mask = Image.open(os.path.join(building_mask_dir, building_image_file)).convert("L")

            try:
                # most decisive line, later on we will use this function to modify the images within the data loader in the training loop
                modified_target_image, modified_target_mask = modify_images(source_image, source_mask, target_image, target_mask)

                # modified_target_image.save(os.path.join(destination_dir, "modified_target_image.png"))
                # modified_target_mask.save(os.path.join(destination_dir, "modified_target_mask.png"))

                modified_target_image.show()
                # modified_target_mask.show()

            except ValueError as e:
                print(e)

    end = time.time()
    print("Time taken: ", end - start)