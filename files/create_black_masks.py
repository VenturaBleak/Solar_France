import os
from PIL import Image
import numpy as np

def create_black_masks(image_dir, mask_dir):
    # Ensure that the mask directory exists
    os.makedirs(mask_dir, exist_ok=True)

    # Get a list of all image files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # For each image file
    for image_file in image_files:
        # Open the image to get its size
        image_path = os.path.join(image_dir, image_file)
        with Image.open(image_path) as img:
            width, height = img.size

        # Create an all-black mask of the same size
        mask = Image.fromarray(np.zeros((height, width), dtype=np.uint8))

        # Save the mask to the mask directory with the same name as the image file
        mask_path = os.path.join(mask_dir, image_file)
        mask.save(mask_path)

if __name__ == '__main__':
    # Usage
    parent_dir = os.path.dirname(os.getcwd())
    building_image_dir = os.path.join(parent_dir, 'data', 'Munich_rooftops_noPV', 'images')
    black_mask_dir = os.path.join(parent_dir, 'data', 'Munich_rooftops_noPV', 'black_masks')

    create_black_masks(building_image_dir, black_mask_dir)