import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_gradient(gradient, parameter_name, epoch, model_name, model_path):
    """
    This function plots the gradient and saves the plot as an image file.
    The y-axis represents the gradient value for a given model parameter.
    If the parameter's gradient is multi-dimensional (such as those in convolutional layers),
    this function takes the mean (or could use sum, or some other aggregation function)
    across all dimensions to derive a scalar value for that parameter. Hence,
    each value on the y-axis corresponds to a scalar representation of a parameter's gradient.

    Please note that the x-axis in this plot is merely an index for the gradient values of
    the specific parameter. It does not represent a meaningful quantity in this context.
    The valuable information here is the gradient value itself represented on the y-axis.

    Parameters
    ----------
    gradient : dict
        The gradients dictionary.
    parameter_name : str
        The name of the parameter for which the gradient is being plotted.
    epoch : int
        The current training epoch.
    model_name : str
        The name of the model being trained.
    model_path : str
        The path where the model is stored.
    """

    # Create a plot.
    fig, ax = plt.subplots()

    # Get the gradient for the parameter
    param_gradient = gradient[parameter_name]

    # If gradient is multi-dimensional (like in convolutional layers), take the mean or sum
    # along all dimensions to get a scalar value per parameter
    if len(param_gradient.shape) > 1:
        param_gradient = param_gradient.mean(axis=tuple(range(param_gradient.ndim)))

    ax.plot(param_gradient)
    ax.set_title(f"{parameter_name} - Epoch {epoch}")

    # Define the output filename.
    output_filename = os.path.join(model_path,f"{model_name}_Epoch{epoch}_{parameter_name}_gradient_plot.png")

    # plot figure
    # plt.show()

    # Save the plot as an image file.
    fig.savefig(output_filename)

    # Close the plot to free up memory.
    plt.close(fig)

def create_gradient_plots(model_name, gradient_name_pattern, output_gif_name, model_path, num_epochs, cwd):
    """Load gradients from pickled files, plot them, and create a gif from the plots."""

    # Get list of all files in the model_path
    all_files = os.listdir(os.path.join(model_path))

    # Initialize a list to hold the gradient files
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
        # print(gradient.keys())

        # Plot the gradient for each parameter in the model
        for parameter_name in gradient:
            plot_gradient(gradient, parameter_name, epoch, model_name, model_path)

if __name__ == "__main__":
    NUM_EPOCHS = 200
    model_folder = "Experiment4"
    model_name = "UNet_ALL"
    cwd = os.getcwd()
    model_path = os.path.join(os.path.dirname(cwd),"models", model_folder)

    # Create gradient GIF
    gradient_name_pattern = model_name + "_Epoch(\d{1,3})_gradients.pkl"
    output_gif_name = model_name + "_gradients" + "_GIF.gif"
    create_gradient_plots(model_name, gradient_name_pattern, output_gif_name, model_path, NUM_EPOCHS, cwd)
