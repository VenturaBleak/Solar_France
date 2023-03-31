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
    x, y = location
    cropped_image = image.crop((y - size // 2, x - size // 2, y + size // 2, x + size // 2))
    cropped_mask = mask.crop((y - size // 2, x - size // 2, y + size // 2, x + size // 2))

    return cropped_image, cropped_mask

def paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, location):
    """Paste the solar panel onto the target image and mask."""
    x, y = location
    cropped_mask_l = cropped_mask.convert("L")
    cropped_mask_rgba = ImageOps.colorize(cropped_mask_l, (0, 0, 0, 0), (255, 255, 255, 255)).convert("RGBA")

    target_image.paste(cropped_image, (y - cropped_image.size[1] // 2, x - cropped_image.size[0] // 2),
                       mask=cropped_mask_rgba)

    cropped_mask_transparent = cropped_mask_rgba.copy()
    cropped_mask_transparent.putalpha(cropped_mask_l)
    target_mask.paste(cropped_mask_transparent,
                      (y - cropped_mask_transparent.size[1] // 2, x - cropped_mask_transparent.size[0] // 2),
                      mask=cropped_mask_transparent)

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

    target_mask_np = np.array(target_mask, dtype=np.float32)
    target_mask_np[target_mask_np > 0] = 255
    target_mask = Image.fromarray(target_mask_np)

    # Find the solar panel in the source mask
    solar_panel_location = find_solar_panel(source_mask_np)

    if solar_panel_location is not None:
        # Crop the solar panel from the source image and source mask
        cropped_image, cropped_mask = crop_solar_panel(source_image, source_mask, solar_panel_location)

        # Paste the solar panel onto the target image and target mask
        paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, solar_panel_location)

        # Save the modified target image and target mask
        target_image.save(os.path.join(image_dir, "modified_target_image.png"))
        target_mask.convert('L').save(os.path.join(mask_dir, "modified_target_mask.png"))

        # Display the modified target image and target mask
        target_image.show()
        target_mask.show()

    else:
        print("No solar panel found in the source mask.")