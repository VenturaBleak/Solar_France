import os
import matplotlib.pyplot as plt
import re
import pickle
import numpy as np
import seaborn as sns
import math

def plot_gradient_stats(model_name, layer_name, model_path, num_epochs):
    """
    This function opens the gradients saved as a pickle file for each epoch, extracts the gradients for
    the specified layer, calculates the mean and standard deviation of all nodes in that layer, and repeats
    this process for all epochs. The mean and std values are then plotted in a line graph.
    """
    # Initialize lists to hold the means, stds and actual_epochs
    means = []
    stds = []
    actual_epochs = []  # List to hold the epochs for which we have gradient data

    # Define the gradient name pattern
    gradient_name_pattern = model_name + "_Epoch(\d{1,3})_gradients.pkl"

    # Get list of all files in the model_path
    all_files = os.listdir(os.path.join(model_path))

    # Initialize a dict to hold the gradient files
    gradient_files = {}

    # Loop over all files and match the filenames against the pattern
    for filename in all_files:
        match = re.match(gradient_name_pattern, filename)
        if match:
            epoch_num = int(match.group(1))  # get epoch number from match group
            gradient_files[epoch_num] = os.path.join(model_path, filename)

    # Loop over the gradient files sorted by epoch
    for epoch in sorted(gradient_files.keys()):  # sort to maintain order
        # Load the gradient
        with open(gradient_files[epoch], 'rb') as f:
            gradient = pickle.load(f)

        # Get the gradient for the specified layer
        layer_gradient = gradient.get(layer_name, None)
        if layer_gradient is None:
            continue

        # Compute the mean and std of the layer gradient
        mean = np.nanmean(layer_gradient)
        std = np.nanstd(layer_gradient)

        # If mean or std is nan, replace it with 0
        if math.isnan(mean):
            mean = 0
        if math.isnan(std):
            std = 0

        # Print for debugging
        print(f'Processing epoch {epoch}: mean={mean}, std={std}')

        # Append the mean and std to the respective lists
        means.append(mean)
        stds.append(std)
        actual_epochs.append(epoch)  # Add the epoch to the list

    # Compute the lower and upper bounds
    lower_bounds = [mean - std for mean, std in zip(means, stds)]
    upper_bounds = [mean + std for mean, std in zip(means, stds)]

    # Use seaborn for better aesthetics
    sns.set(style="white")

    # Create a plot
    plt.figure(figsize=(10, 6))

    plt.plot(actual_epochs, means, label='Mean', color='blue')
    plt.fill_between(actual_epochs, lower_bounds, upper_bounds, color='blue', alpha=0.2)

    plt.title(f'Gradient Statistics for {layer_name} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    NUM_EPOCHS = 200
    model_folder = "Experiment1"

    cwd = os.getcwd()
    model_path = os.path.join(os.path.dirname(cwd),"models", model_folder)

    # Name of the layer for which the gradient stats will be plotted
    model_name = "UNet_pretrained100"
    layer_name = 'downs.3.conv.4.weight'

    # model_name = "B2_All_BCE"
    # layer_name = 'mit.stages.3.2.2.1.fn.net.1.net.0.bias'

    # Call the function to plot the gradient stats
    plot_gradient_stats(model_name, layer_name, model_path, NUM_EPOCHS)