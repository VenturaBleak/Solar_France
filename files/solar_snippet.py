import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw

def find_solar_panel(mask):
    """Find the solar panel in the mask."""
    white_pixels = np.argwhere(mask == 255)

    if len(white_pixels) > 0:
        return white_pixels[0]
    else:
        return None

def crop_solar_panel(image, mask, location, size=50):
    """Crop the solar panel from the image and mask."""
    mask_np = np.array(mask, dtype=np.float32)

    assert np.sum(mask_np == 255) > 0, "No white pixels in mask."

    white_pixels = np.argwhere(mask_np == 255)

    if white_pixels.size == 0:
        return None, None

    y_min, x_min = np.min(white_pixels, axis=0)
    y_max, x_max = np.max(white_pixels, axis=0)

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    return cropped_image, cropped_mask


def paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, location):
    """Paste the solar panel onto the target image and mask."""
    x, y = location
    cropped_mask_l = cropped_mask.convert("L")
    cropped_mask_rgba = Image.new("RGBA", cropped_mask_l.size)
    ImageDraw.Draw(cropped_mask_rgba).bitmap((0, 0), cropped_mask_l, fill=(255, 255, 255, 255))

    target_image.paste(cropped_image, (y - cropped_image.size[1] // 2, x - cropped_image.size[0] // 2), mask=cropped_mask_rgba)
    target_mask.paste(cropped_mask_l, (y - cropped_mask_l.size[1] // 2, x - cropped_mask_l.size[0] // 2), mask=cropped_mask_l)

if __name__ == "__main__":
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # Define the image and mask directories under the parent directory
    image_dir = os.path.join(parent_dir, 'data', 'trial', 'images')
    mask_dir = os.path.join(parent_dir, 'data', 'trial', 'masks')

    # Load source image, source mask, target image, and target mask
    source_image = Image.open(os.path.join(image_dir, "AABCN78E1YHCWW.png")).convert("RGB")
    source_mask = Image.open(os.path.join(mask_dir, "AABCN78E1YHCWW.png")).convert("L")
    target_image = Image.open(os.path.join(image_dir, "AAJKM9CB2TTSEF.png")).convert("RGB")
    target_mask = Image.open(os.path.join(mask_dir, "AAJKM9CB2TTSEF.png")).convert("L")

    # Convert masks to binary masks
    source_mask_np = np.array(source_mask, dtype=np.float32)
    source_mask_np[source_mask_np > 0] = 255
    source_mask = Image.fromarray(source_mask_np)
    # assert
    assert np.sum(source_mask_np == 255) > 0, "No white pixels in source mask."

    target_mask_np = np.array(target_mask, dtype=np.float32)
    target_mask_np[target_mask_np > 0] = 255
    target_mask = Image.fromarray(target_mask_np)

    # Find the solar panel in the source mask
    solar_panel_location = find_solar_panel(source_mask_np)

    if solar_panel_location is not None:
        # Crop the solar panel from the source image and source mask
        cropped_image, cropped_mask = crop_solar_panel(source_image, source_mask, solar_panel_location)

        if cropped_image is not None and cropped_mask is not None:
            # Paste the solar panel onto the target image and target mask
            paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, solar_panel_location)


            # Save the modified target image and target mask
            target_image.save(os.path.join(image_dir, "modified_target_image.png"))
            target_mask.convert('L').save(os.path.join(mask_dir, "modified_target_mask.png"))

            # Display the modified target image and target mask
            target_image.show()
            target_mask.show()
        else:
            print("No white pixels found in the source mask.")
    else:
        print("No solar panel found in the source mask.")