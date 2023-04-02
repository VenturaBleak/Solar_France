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
        # print(f"Removed: {image_file}")

    # Find mask files without corresponding image files
    unmatched_masks = masks_set - images_set
    for mask_file in sorted(unmatched_masks):
        os.remove(os.path.join(mask_dir, mask_file))
        # print(f"Removed: {mask_file}")
        files_removed += 1

    print(f"Removed {files_removed} files.")

if __name__ == '__main__':
    # run data_cleaning.py on the google images and masks, using cwd
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)

    image_dir = os.path.join(parent_dir, 'data', 'bdappv', 'google', 'images')
    mask_dir = os.path.join(parent_dir, 'data', 'bdappv', 'google', 'masks')

    remove_unmatched_files(image_dir, mask_dir)
