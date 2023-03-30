import os
from PIL import Image

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def process_all_images(image_dir, mask_dir, image_list, mask_list):
    prev_image_width, prev_image_height = None, None
    prev_mask_width, prev_mask_height = None, None

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

    # if assertions do not fail, print task is complete
    print("All image and mask sizes match.")

def main():
    image_dir = os.path.join(os.getcwd(), 'data', 'trial', 'masks')
    mask_dir = os.path.join(os.getcwd(), 'data',  'trial', 'masks')

    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))

    process_all_images(image_dir, mask_dir, images, masks)

if __name__ == "__main__":
    main()