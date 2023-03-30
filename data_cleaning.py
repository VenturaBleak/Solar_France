import os

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
        print(f"Removed: {image_file}")

    # Find mask files without corresponding image files
    unmatched_masks = masks_set - images_set
    for mask_file in sorted(unmatched_masks):
        os.remove(os.path.join(mask_dir, mask_file))
        print(f"Removed: {mask_file}")
        files_removed += 1

    print(f"Removed {files_removed} files.")

# Get the current working directory
cwd = os.getcwd()

# Define the image and mask directories under the current working directory
image_dir = os.path.join(cwd, 'data', 'trial', 'images')
mask_dir = os.path.join(cwd, 'data', 'trial', 'masks')

remove_unmatched_files(image_dir, mask_dir)