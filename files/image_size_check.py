import os
from PIL import Image
import shutil

######################################################################################################################
# Helper functions
######################################################################################################################

#######################
# Check Image dimensions
#######################

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def check_dimensions(image_dir, mask_dir, image_list, mask_list):
    prev_image_width, prev_image_height = None, None
    prev_mask_width, prev_mask_height = None, None

    # Initialize the variables with default values
    image_width, image_height = 0, 0
    mask_width, mask_height = 0, 0

    count = 0

    for image_name, mask_name in zip(image_list, mask_list):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        image_width, image_height = get_image_size(image_path)
        mask_width, mask_height = get_image_size(mask_path)

        if prev_image_width and prev_image_height:
            assert prev_image_width == image_width and prev_image_height == image_height, f"Image sizes do not match:{image_path}"
        if prev_mask_width and prev_mask_height:
            assert prev_mask_width == mask_width and prev_mask_height == mask_height, f"Mask sizes do not match:{mask_path}"


        prev_image_width, prev_image_height = image_width, image_height
        prev_mask_width, prev_mask_height = mask_width, mask_height

        # print(f"Image: {image_name}, width: {image_width}, height: {image_height}")
        # print(f"Mask: {mask_name}, width: {mask_width}, height: {mask_height}")

        count += 1

    # if assertions do not fail, print task is complete
    print(f"All image and mask sizes match. {count} files checked -> Image size: {image_width}x{image_height}. Mask size: {mask_width}x{mask_height}.")

#######################
# Remove Unmatched Files
#######################

def remove_unmatched_files(image_dir, mask_dir):
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    files_removed = 0

    images_set = set(images)
    masks_set = set(masks)

    # Find image files without corresponding mask files
    unmatched_images = images_set - masks_set
    for image_file in sorted(unmatched_images):
        os.remove(os.path.join(image_dir, image_file))
        # print(f"Removed: {image_file}")

    # Find mask files without corresponding image files
    unmatched_masks = masks_set - images_set
    for mask_file in sorted(unmatched_masks):
        os.remove(os.path.join(mask_dir, mask_file))
        # print(f"Removed: {mask_file}")
        files_removed += 1

    print(f"Removed {files_removed} files from {image_dir} and {mask_dir}.")

#######################
# Check Positive Masks
#######################

def check_positive_masks_move_to_negative(images_positive_dir, masks_positive_dir, images_negative_dir,
                                          masks_negative_dir):
    positive_masks = sorted(os.listdir(masks_positive_dir))

    count = 0

    for mask_file in positive_masks:
        mask_path = os.path.join(masks_positive_dir, mask_file)
        mask = Image.open(mask_path)

        white_pixels = False
        # Check if mask has any white pixels
        for pixel in mask.getdata():
            if pixel == 255:
                white_pixels = True
                break

        if not white_pixels:
            # Move mask and corresponding image to negative folders
            image_file = mask_file  # Assuming mask and image have the same name
            image_path = os.path.join(images_positive_dir, image_file)

            shutil.move(image_path, os.path.join(images_negative_dir, image_file))
            shutil.move(mask_path, os.path.join(masks_negative_dir, mask_file))
            print(f"Moved {image_file} and {mask_file} to negative folders")

            count += 1

    print(f"Moved {count} files to negative folders.")

###############################
# Check Negative Masks
###############################
def check_negative_masks_move_to_positive(images_positive_dir, masks_positive_dir, images_negative_dir, masks_negative_dir):
    negative_masks = sorted(os.listdir(masks_negative_dir))

    count = 0

    for mask_file in negative_masks:
        mask_path = os.path.join(masks_negative_dir, mask_file)
        mask = Image.open(mask_path)

        white_pixels = False
        for pixel in mask.getdata():
            if pixel == 255:
                white_pixels = True
                break

        if white_pixels:
            # Move mask and corresponding image to positive folders
            image_file = mask_file  # Assuming mask and image have the same name
            image_path = os.path.join(images_negative_dir, image_file)

            shutil.move(image_path, os.path.join(images_positive_dir, image_file))
            shutil.move(mask_path, os.path.join(masks_positive_dir, mask_file))
            print(f"Moved {image_file} and {mask_file} to positive folders")

            count += 1

    print(f"Moved {count} files to positive folders.")

###############################
# Remove Images with Less than X White Pixels
###############################

def count_white_pixels(mask):
    return sum(pixel == 255 for pixel in mask.getdata())

def remove_images_with_less_than_x_white_pixels(image_dir, mask_dir, x=10):
    masks = sorted(os.listdir(mask_dir))
    images = sorted(os.listdir(image_dir))

    count = 0

    for image_name, mask_name in zip(images, masks):
        mask_path = os.path.join(mask_dir, mask_name)
        mask = Image.open(mask_path)

        if count_white_pixels(mask) < x:
            image_path = os.path.join(image_dir, image_name)
            os.remove(image_path)
            os.remove(mask_path)
            print(f"Removed {image_name} and {mask_name} due to insufficient white pixels")
            count += 1

    print(f"Removed {count} images and masks with less than {x} white pixels.")


############################################################################################################
# Main functions
############################################################################################################

def check_dimensions_main(dataset_folders, data_dir):
    print("###############################################")
    print("Checking dimensions in images and masks folders...")

    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    for dataset in dataset_folders:
        print(f"Checking dataset: {dataset}")

        images_positive_dir, images_negative_dir, masks_positive_dir, masks_negative_dir = get_directory_paths(dataset, data_dir)

        images_positive_list = sorted(os.listdir(images_positive_dir))
        images_negative_list = sorted(os.listdir(images_negative_dir))
        masks_positive_list = sorted(os.listdir(masks_positive_dir))
        masks_negative_list = sorted(os.listdir(masks_negative_dir))

        print("Checking dimensions in images_positive and masks_positive folders...")
        check_dimensions(images_positive_dir, masks_positive_dir, images_positive_list, masks_positive_list)
        print("Checking dimensions in images_negative and masks_negative folders...")
        check_dimensions(images_negative_dir, masks_negative_dir, images_negative_list, masks_negative_list)
        print()

def remove_unmatched_files_main(dataset_folders, data_dir):
    print("###############################################")
    print("Removing unmatched files in images and masks folders...")

    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    for dataset in dataset_folders:
        print(f"Removing unmatched files in dataset: {dataset}")

        images_positive_dir, images_negative_dir, masks_positive_dir, masks_negative_dir = get_directory_paths(dataset, data_dir)

        remove_unmatched_files(images_positive_dir, masks_positive_dir)
        remove_unmatched_files(images_negative_dir, masks_negative_dir)
        print()
def check_positive_masks_move_to_negative_main(dataset_folders, data_dir):
    print("###############################################")
    print("Checking positive masks and moving to negative if necessary...")

    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    for dataset in dataset_folders:
        print(f"Checking positive masks and moving to negative if necessary in dataset: {dataset}")

        images_positive_dir, images_negative_dir, masks_positive_dir, masks_negative_dir = get_directory_paths(dataset, data_dir)

        check_positive_masks_move_to_negative(images_positive_dir, masks_positive_dir, images_negative_dir, masks_negative_dir)
        print()

def check_negative_masks_move_to_positive_main(dataset_folders, data_dir):
    print("###############################################")
    print("Checking negative masks and moving to positive if necessary...")

    for dataset in dataset_folders:
        print(f"Checking negative masks and moving to positive if necessary in dataset: {dataset}")

        images_positive_dir, images_negative_dir, masks_positive_dir, masks_negative_dir = get_directory_paths(dataset, data_dir)

        check_negative_masks_move_to_positive(images_positive_dir, masks_positive_dir, images_negative_dir, masks_negative_dir)
        print()

def remove_images_with_less_than_x_white_pixels_main(dataset_folders, data_dir):
    # remove images with less than x white pixels
    print("###############################################")
    print("Removing images and masks with less than x white pixels...")
    for dataset in dataset_folders:
        images_positive_dir, images_negative_dir, masks_positive_dir, masks_negative_dir = get_directory_paths(dataset, data_dir)

        print(f"Processing dataset: {dataset}")
        print("Processing positive images and masks...")
        remove_images_with_less_than_x_white_pixels(images_positive_dir, masks_positive_dir)
        print()

def get_directory_paths(dataset, data_dir):
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)

    images_positive_dir = os.path.join(parent_dir, data_dir, dataset, "images_positive")
    images_negative_dir = os.path.join(parent_dir, data_dir, dataset, "images_negative")
    masks_positive_dir = os.path.join(parent_dir, data_dir, dataset, "masks_positive")
    masks_negative_dir = os.path.join(parent_dir, data_dir, dataset, "masks_negative")

    return images_positive_dir, images_negative_dir, masks_positive_dir, masks_negative_dir

# New function to remove mask and corresponding image when there is no single white pixel on the mask
def remove_non_white_masks(image_dir, mask_dir):
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))

    count = 0

    for image_name, mask_name in zip(images, masks):
        mask_path = os.path.join(mask_dir, mask_name)
        mask = Image.open(mask_path)

        white_pixels = False
        for pixel in mask.getdata():
            if pixel == 255:
                white_pixels = True
                break

        if not white_pixels:
            image_path = os.path.join(image_dir, image_name)
            os.remove(image_path)
            os.remove(mask_path)
            print(f"Removed {image_name} and {mask_name}")
            count += 1

    print(f"Removed {count} files without white pixels.")

def check_dataset_no_pv(dataset, data_dir="data"):
    print("###############################################")
    print(f"Processing dataset: {dataset}")

    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)

    images_dir = os.path.join(parent_dir, data_dir, dataset, "images")
    masks_dir = os.path.join(parent_dir, data_dir, dataset, "building_masks")

    images_list = sorted(os.listdir(images_dir))
    masks_list = sorted(os.listdir(masks_dir))

    print("Checking dimensions in images and building_masks folders...")
    check_dimensions(images_dir, masks_dir, images_list, masks_list)

    print("Removing unmatched files in images and building_masks folders...")
    remove_unmatched_files(images_dir, masks_dir)

    print("Removing masks and corresponding images without white pixels...")
    remove_non_white_masks(images_dir, masks_dir)
    print()


if __name__ == "__main__":

    # process the dataset without PV
    dataset_no_pv = "Munich_rooftops_noPV"
    check_dataset_no_pv(dataset_no_pv)


    # specify the dataset folders to check
    dataset_folders = ['Munich']

    data_dir = "data"

    # perform check dimensions
    check_dimensions_main(dataset_folders, data_dir)

    # perform remove unmatched files
    remove_unmatched_files_main(dataset_folders, data_dir)

    # perform check positive masks and move to negative
    check_positive_masks_move_to_negative_main(dataset_folders, data_dir)

    # perform check negative masks and move to positive
    check_negative_masks_move_to_positive_main(dataset_folders, data_dir)

    data_dir = "data_train_aug"

    # remove images with less than x white pixels
    remove_images_with_less_than_x_white_pixels_main(dataset_folders, data_dir)

    data_dir = "data_test_aug"

    # remove images with less than x white pixels
    remove_images_with_less_than_x_white_pixels_main(dataset_folders, data_dir)