import os
import shutil
import random

def split_data(dataset_fractions, parent_dir, data_dir, train_fraction):
    """
    Split the datasets into training, validation, and testing sets
    by copying the files from the original datasets to different directories.

    :param dataset_fractions: list of datasets to be split.
    :param parent_dir: parent directory where the datasets are stored.
    """
    # Loop over each dataset
    for ds_info in dataset_fractions:
        dataset_name, pos_fraction, neg_fraction = ds_info

        # Loop over each category: positive and negative
        for category in ['positive', 'negative']:
            # Define the directories for images and masks in the original dataset
            image_dir = os.path.join(parent_dir, data_dir, dataset_name, f'images_{category}')
            mask_dir = os.path.join(parent_dir, data_dir, dataset_name, f'masks_{category}')

            # Get the list of files in the image directory
            files = os.listdir(image_dir)

            # Shuffle the files to ensure a random split
            random.shuffle(files)

            # randomly sample
            if category == 'positive':
                files = files[:int(len(files) * pos_fraction)]
            elif category == 'negative':
                files = files[:int(len(files) * neg_fraction)]

            # Calculate the number of files for training and validation
            num_train = int(len(files) * train_fraction)
            num_val = int(len(files) * 0)

            # Split the files into train, validation, and test sets
            files_train, files_val, files_test = files[:num_train], files[num_train:num_train + num_val], files[num_train + num_val:]

            # Loop over each split: train, val, test
            for split, files_split in [('train', files_train), ('val', files_val), ('test', files_test)]:
                # Define the directories for images and masks in the split dataset
                split_image_dir = os.path.join(parent_dir, f'data_{split}', dataset_name, f'images_{category}')
                split_mask_dir = os.path.join(parent_dir, f'data_{split}', dataset_name, f'masks_{category}')

                # Create the directories if they don't exist
                os.makedirs(split_image_dir, exist_ok=True)
                os.makedirs(split_mask_dir, exist_ok=True)

                # Copy the files from the original dataset to the split dataset
                for file in files_split:
                    shutil.copy(os.path.join(image_dir, file), os.path.join(split_image_dir, file))
                    shutil.copy(os.path.join(mask_dir, file), os.path.join(split_mask_dir, file))

                # delete folder, if it contains no files
                if len(os.listdir(split_image_dir)) == 0:
                    os.rmdir(split_image_dir)
                if len(os.listdir(split_mask_dir)) == 0:
                    os.rmdir(split_mask_dir)

    # loop over each dataset and delete folder, if it contains no files
    for ds_info in dataset_fractions:
        dataset_name, _, _ = ds_info
        for split in ['train', 'val', 'test']:
            dataset_dir = os.path.join(parent_dir, f'data_{split}', dataset_name)
            # if image dir does not exist, skip
            if os.path.exists(dataset_dir) == True:
                if len(os.listdir(dataset_dir)) == 0:
                    os.rmdir(dataset_dir)

def test_no_overlap_and_count(dataset_fractions, parent_dir):
    """
    Unit test to verify:
    1. No files are missing in the split datasets (training, validation, testing) compared to the original dataset.
    2. No extra files are present in the split datasets compared to the original dataset.
    3. No overlap (data leakage) between the split datasets.

    :param dataset_fractions: list of datasets to be tested.
    :param parent_dir: parent directory where the datasets are stored.
    """
    from itertools import combinations

    # Loop over each dataset
    for ds_info in dataset_fractions:
        dataset_name, _, _ = ds_info

        # Initialize lists to store filepaths
        original_files = []
        split_files = []
        split_filepaths = {'train': [], 'val': [], 'test': []}  # Store filepaths for each split

        # Loop over each category: positive and negative
        for category in ['positive', 'negative']:
            # Define the directories for images and masks in the original dataset
            original_image_dir = os.path.join(parent_dir, 'data', dataset_name, f'images_{category}')
            original_mask_dir = os.path.join(parent_dir, 'data', dataset_name, f'masks_{category}')

            # Extend the list of original_files with the filepaths of the images and masks
            original_files.extend(
                [os.path.join(dataset_name, f'images_{category}', f) for f in os.listdir(original_image_dir)])
            original_files.extend(
                [os.path.join(dataset_name, f'masks_{category}', f) for f in os.listdir(original_mask_dir)])

            # Loop over each split: train, val, test
            for split in ['train', 'val', 'test']:
                # Define the directories for images and masks in the split dataset
                split_image_dir = os.path.join(parent_dir, f'data_{split}', dataset_name, f'images_{category}')
                split_mask_dir = os.path.join(parent_dir, f'data_{split}', dataset_name, f'masks_{category}')

                # Extend the list of split_files with the filepaths of the images and masks
                split_files_subset = [os.path.join(dataset_name, f'images_{category}', f) for f in
                                      os.listdir(split_image_dir)]
                split_files_subset.extend(
                    [os.path.join(dataset_name, f'masks_{category}', f) for f in os.listdir(split_mask_dir)])

                # Update split_files and split_filepaths
                split_files.extend(split_files_subset)
                split_filepaths[split].extend(split_files_subset)

        # Assert no files are missing in the split datasets
        assert len(set(original_files).difference(
            set(split_files))) == 0, f'Some files in {dataset_name} are missing in split datasets.'

        # Assert no extra files are present in the split datasets
        assert len(set(split_files).difference(
            set(original_files))) == 0, f'Some files in split datasets of {dataset_name} are not in original dataset.'

        # Check for overlaps between each pair of datasets (train, val, test)
        for split1, split2 in combinations(['train', 'val', 'test'], 2):
            assert len(set(split_filepaths[split1]).intersection(set(split_filepaths[
                                                                         split2]))) == 0, f"Overlap found between {split1} and {split2} datasets of {dataset_name}."

    print("No overlap found between training, validation, and testing datasets.")

if __name__ == '__main__':
    # Get the current working directory
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    train_fraction = 0.8

    data_dir = "data"
    dataset_fractions = [
        ['France_google', 0, 0],
        ['France_ign', 0, 0],
        ['Munich', 0, 0],
        ['China', 1, 1],
        ['Denmark', 1, 0]
    ]

    data_dir = "data_NL"
    dataset_fractions = [
        ['Heerlen_2018_HR_output', 1, 0],
        ['ZL_2018_HR_output', 1, 0],
    ]

    # Create the splits
    split_data(dataset_fractions, parent_dir, data_dir, train_fraction)

    # Unit-Test the splits
    # test_no_overlap_and_count(dataset_fractions, parent_dir)