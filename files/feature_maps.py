import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import DoubleConv
import os

def visualize_feature_maps(model, img_path, train_mean, train_std, file_name, folder, device="cpu", img_height=416, img_width=416):
    """
    Visualize the feature maps of the model. This function loads an image, transforms it, feeds it through
    the convolutional layers of the model, and then plots the resulting feature maps.

    Parameters:
    model: PyTorch model whose feature maps are to be visualized.
    img_path: Path to the input image.
    train_mean: Mean of the training set, used for normalization.
    train_std: Standard deviation of the training set, used for normalization.
    device: Device to perform computations on. Default is CPU.
    img_height: The height to resize the input image to.
    img_width: The width to resize the input image to.
    """
    PLT_MULT_LAYERS = True
    PLT_KERNEL = True
    PLT_AGGREGATED_LAYERS = True
    SHOW = False

    model.eval()
    with torch.no_grad():
        # Set device to cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # show the image
        if SHOW == True:
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        # Display the loaded image
        plt.savefig(os.path.join(folder, f"{file_name}_fm_original.png"), bbox_inches='tight', pad_inches=0)

        # Define image transformations: resize, convert to tensor, and normalize
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
        ])

        # Apply transformations to the image, add an extra dimension to fit the model's input size,
        # and send the image tensor to the specified device
        image = transform(image).unsqueeze(0).to(device)

        # Initialize a list to store the convolutional layers of the model
        conv_layers = []
        # Iterate over the children of the downsampling part of the model
        for child in list(model.downs):
            # Iterate over the layers in the double convolution block
            for layer in child.conv:
                # If the layer is a 2D convolutional layer, add it to the list
                if type(layer) == torch.nn.modules.conv.Conv2d:
                    conv_layers.append(layer)

        # print(f"conv_layers: {conv_layers}")

        if PLT_MULT_LAYERS:
            # Initialize a list to store the feature maps of each layer
            featuremaps = [conv_layers[0](image)]
            for layer_number in range(1, len(conv_layers)):
                # Compute the feature map of the current layer by applying it to the feature map of the previous layer
                featuremaps.append(conv_layers[layer_number](featuremaps[-1]))

            # Visualize every second feature map
            for layer_number in range(0, len(featuremaps), 2):  # step size of 2 to plot every second feature map
                fig = plt.figure(figsize=(30, 30))
                # Detach the feature maps from the computation graph and get the first image's feature maps
                layers = featuremaps[layer_number][0, :, :, :].detach()
                # Iterate over each feature map
                for i, filter in enumerate(layers):
                    # Plot only the first 64 feature maps
                    if i == 64:
                        break
                    plt.subplot(8, 8, i + 1)  # plot in an 8x8 grid
                    plt.imshow(filter.cpu(), cmap='gray')  # plot the feature map in grayscale
                    plt.axis('off')  # turn off axis labels
                    plt.title(f'Feature map {i + 1}', fontsize=20)
                fig.suptitle(f'Feature maps - Block {int(layer_number / 2) + 1}', fontsize=100)

                if SHOW == True:
                    plt.show()

                # Save the figure as an image file
                image_file_path = os.path.join(folder, f"{file_name}_fm_mult_layers_{layer_number}.png")
                fig.savefig(image_file_path, bbox_inches='tight')

        if PLT_KERNEL:
            # Visualize the weights
            for layer_number, layer in enumerate(conv_layers):
                if layer_number % 2 == 0:  # visualize every second layer
                    fig = plt.figure(figsize=(30, 30))
                    weights = layer.weight.detach()  # detach the weights from the computation graph
                    for i, weight in enumerate(weights):
                        if i == 64:
                            break
                        plt.subplot(8, 8, i + 1)
                        plt.imshow(weight[0], cmap='gray')  # visualize the first channel of the weight
                        plt.axis('off')
                        plt.title(f'Filter {i + 1}', fontsize=20)
                    fig.suptitle(f'Filters - Block {int(layer_number / 2) + 1}', fontsize=100)

                    if SHOW == True:
                        plt.show()

                    # Save the figure as an image file
                    image_file_path = os.path.join(folder, f"{file_name}_fm_kernel_{layer_number/2+1}.png")
                    fig.savefig(image_file_path, bbox_inches='tight')

        if PLT_AGGREGATED_LAYERS:
            # Initialize empty lists for storing outputs from each convolutional layer and the names of the layers
            outputs = []
            names = []

            # Pass the image through each convolutional layer
            for layer in conv_layers:
                # Apply the current convolutional layer to the image. The output is a feature map.
                image = layer(image)

                # Add the output feature map to the list
                outputs.append(image)

                # Add the name of the layer to the list. This is used for visualization later.
                names.append(str(layer))

            # Initialize an empty list for storing processed (grayscaled) feature maps
            processed = []

            # Process each output feature map to convert it to grayscale
            for feature_map in outputs:
                # Remove the batch dimension from the feature map using the squeeze() function.
                # The output feature map is a 4D tensor with shape (batch_size, channels, height, width).
                # After squeezing, it becomes a 3D tensor with shape (channels, height, width).
                feature_map = feature_map.squeeze(0)

                # Convert the 3D tensor to 2D by summing the same element of every channel.
                # This results in a grayscale image where each pixel's value is the sum of that pixel's values across all channels.
                gray_scale = torch.sum(feature_map, 0)

                # Normalize the grayscale image by dividing it by the number of channels.
                # This is done to keep the pixel values in a reasonable range.
                gray_scale = gray_scale / feature_map.shape[0]

                # Convert the grayscale image from a PyTorch tensor to a NumPy array and add it to the list.
                # This is done because matplotlib, which is used for visualization, works with NumPy arrays.
                processed.append(gray_scale.data.cpu().numpy())

            # Create a new figure for visualization
            fig = plt.figure(figsize=(25, 15))

            # Visualize each processed feature map
            for i in range(len(processed)):
                # Add a subplot to the figure for the current feature map
                a = fig.add_subplot(2, 4, i + 1)

                # Display the grayscale feature map in the subplot
                imgplot = plt.imshow(processed[i])

                # Turn off axis labels because they are not relevant in this case
                a.axis("off")

                # Add a title to the subplot. The title is the number of the layer in this case.
                a.set_title(f"Block {int(i/2)+1} - Layer {(i%2)+1}", fontsize=30)

            # Add a title to the figure
            fig.suptitle('Feature maps', fontsize=100)

            if SHOW == True:
                plt.show()

            # Save the figure as an image file
            image_file_path = os.path.join(folder, f"{file_name}_fm_aggregated.png")
            fig.savefig(image_file_path, bbox_inches='tight')

    model.train()