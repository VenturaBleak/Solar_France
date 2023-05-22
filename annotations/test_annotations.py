import cv2
import matplotlib.pyplot as plt

# Test 1: Visually inspect some of your masks
def test_mask_overlay(image_file, mask_file):
    # Load the image and mask
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file, 0)  # load in grayscale mode

    # Create an overlay by adding the image and mask
    overlay = cv2.addWeighted(image, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)

    # Display the overlay
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.show()

    # save the overlay under the name of the mask + _overlay.png
    cv2.imwrite(mask_file[:-4] + '_overlay.png', overlay)


# Test 2: Check the area of the objects in the mask
def test_mask_area(mask_file, expected_area):
    # Load the mask
    mask = cv2.imread(mask_file, 0)  # load in grayscale mode

    # Calculate the area of the objects in the mask (number of white pixels)
    area = np.sum(mask == 255)

    # Assert that the area matches the expected area
    assert area == expected_area, f"Expected area of {expected_area}, but got {area}"


# Call the tests
test_mask_overlay('09a0161d-0aa8-43cb-83f3-7eac02d8bb76_rgb_hr_20181.png', r'09a0161d-0aa8-43cb-83f3-7eac02d8bb76_rgb_hr_2018.png')
test_mask_overlay('09b09f45-21c2-442f-aa95-aa05c65ef0dc_rgb_hr_20181.png', r'09b09f45-21c2-442f-aa95-aa05c65ef0dc_rgb_hr_2018.png')

# test_mask_area('00bd7de8-3272-4475-bec0-33cf2e2042e6_rgb_hr_2018.png_mask.png', expected_area)