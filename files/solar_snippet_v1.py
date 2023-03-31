import os
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageOps, ImageDraw

class NoWhitePixelsException(Exception):
    pass

class NoSolarPanelException(Exception):
    pass


def find_largest_white_patch(mask):
    # Find the largest white patch in the mask and its bounding box.

    # Label connected components in the binary mask
    mask_labeled, num_features = ndi.label(mask)

    # If there are no connected components, return None
    if num_features == 0:
        return None, None

    # Calculate the size of each connected component
    patch_sizes = [np.sum(mask_labeled == i) for i in range(1, num_features + 1)]

    # Find the index of the largest connected component
    largest_patch = np.argmax(patch_sizes) + 1

    # Find the pixels belonging to the largest connected component
    largest_patch_pixels = np.argwhere(mask_labeled == largest_patch)

    # Calculate the center of the largest connected component
    center = np.mean(largest_patch_pixels, axis=0).astype(np.int32)

    # Find the minimum and maximum coordinates of the largest connected component
    y_min, x_min = np.min(largest_patch_pixels, axis=0)
    y_max, x_max = np.max(largest_patch_pixels, axis=0)

    # Create a bounding box for the largest connected component
    bounding_box = (x_min, y_min, x_max, y_max)

    # Return the center and bounding box of the largest connected component
    return tuple(center), bounding_box

def find_solar_panel(mask):
    """Find the solar panel in the mask."""

    # Find the coordinates of the white pixels in the mask
    white_pixels = np.argwhere(mask == 255)

    # If there are white pixels, return the first one as the solar panel location
    if len(white_pixels) > 0:
        return white_pixels[0]
    # Otherwise, return None
    else:
        return None

def crop_solar_panel(image, mask, location, size=50):
    """Crop the solar panel from the image and mask."""
    mask_np = np.array(mask, dtype=np.float32)

    if np.sum(mask_np == 255) == 0:
        raise NoWhitePixelsException("No white pixels in mask.")

    # Find connected components
    labeled_mask, num_features = ndi.label(mask_np == 255)

    # Find the largest connected component
    largest_cc = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1

    # Create a binary mask with only the largest connected component
    largest_cc_mask = (labeled_mask == largest_cc).astype(np.uint8) * 255

    white_pixels = np.argwhere(largest_cc_mask == 255)
    y_min, x_min = np.min(white_pixels, axis=0)
    y_max, x_max = np.max(white_pixels, axis=0)

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    return cropped_image, cropped_mask

def paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, location, bounding_box):
    """Paste the solar panel onto the target image and mask within the boundaries of the largest patch."""
    x, y = location
    x_min, y_min, x_max, y_max = bounding_box

    # Ensure the pasting position is within the bounding box
    paste_x = max(min(x - cropped_image.size[0] // 2, x_max - cropped_image.size[0]), x_min)
    paste_y = max(min(y - cropped_image.size[1] // 2, y_max - cropped_image.size[1]), y_min)

    cropped_mask_l = cropped_mask.convert("L")
    cropped_mask_rgba = Image.new("RGBA", cropped_mask_l.size)
    ImageDraw.Draw(cropped_mask_rgba).bitmap((0, 0), cropped_mask_l, fill=(255, 255, 255, 255))

    target_image.paste(cropped_image, (paste_y, paste_x), mask=cropped_mask_rgba)
    target_mask.paste(cropped_mask_l, (paste_y, paste_x), mask=cropped_mask_l)


def clear_original_mask(target_mask):
    """Clear the original mask."""
    target_mask_np = np.array(target_mask)
    target_mask_np.fill(0)
    return Image.fromarray(target_mask_np)

if __name__ == "__main__":
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # Define the image and mask directories under the parent directory
    image_dir = os.path.join(parent_dir, 'data', 'trial', 'images')
    mask_dir = os.path.join(parent_dir, 'data', 'trial', 'masks')

    # Load source image, source mask, target image, and target mask
    source_image = Image.open(os.path.join(image_dir, "ABEKP6A34YFLOO.png")).convert("RGB")
    source_mask = Image.open(os.path.join(mask_dir, "ABEKP6A34YFLOO.png")).convert("L")
    target_image = Image.open(os.path.join(image_dir, "AAYQM71ADIEGMN.png")).convert("RGB")
    target_mask = Image.open(os.path.join(mask_dir, "AAYQM71ADIEGMN.png")).convert("L")

    # Convert masks to binary masks
    source_mask_np = np.array(source_mask, dtype=np.float32)
    source_mask_np[source_mask_np > 0] = 255
    source_mask = Image.fromarray(source_mask_np)
    # assert
    assert np.sum(source_mask_np == 255) > 0, "No white pixels in source mask."

    target_mask_np = np.array(target_mask, dtype=np.float32)
    target_mask_np[target_mask_np > 0] = 255
    target_mask = Image.fromarray(target_mask_np)

    # Clear the original mask of the target image
    target_mask = clear_original_mask(target_mask)

    # Find the solar panel in the source mask
    try:
        solar_panel_location = find_solar_panel(source_mask_np)
        cropped_image, cropped_mask = crop_solar_panel(source_image, source_mask, solar_panel_location)
        largest_patch_location, bounding_box = find_largest_white_patch(target_mask_np)
        paste_solar_panel(target_image, target_mask, cropped_image, cropped_mask, largest_patch_location, bounding_box)


        # Save the modified target image and target mask
        target_image.save(os.path.join(image_dir, "modified_target_image.png"))
        target_mask.convert('L').save(os.path.join(mask_dir, "modified_target_mask.png"))

        # Display the modified target image and target mask
        target_image.show()
        target_mask.show()

    except NoWhitePixelsException as e:
        print(e)
    except NoSolarPanelException as e:
        print(e)