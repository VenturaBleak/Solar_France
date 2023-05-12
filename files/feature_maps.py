import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import DoubleConv

def visualize_feature_maps(model, img_path, train_mean, train_std):
    """Visualize the feature maps of the model
    code adapted from three sources:
    1. https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
    2. https://github.com/Ashborn-SM/Visualizing-Filters-and-Feature-Maps-in-Convolutional-Neural-Networks-using-PyTorch/blob/master/Visualiser.py
    3. https://github.com/Ashborn-SM/Visualizing-Filters-and-Feature-Maps-in-Convolutional-Neural-Networks-using-PyTorch/blob/master/extractor.py

    Purpose: visualize the feature maps of the model

    :param
    model: the model to visualize
    img_path: the path to the image to visualize
    train_mean: the mean of the training set
    train_std: the standard deviation of the training set
    """
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_HEIGHT = 416
    IMAGE_WIDTH = 416

    # Load and transform the image
    image = Image.open(img_path).convert("RGB")
    plt.imshow(image)

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Append all the conv layers to a list
    conv_layers = []
    for child in list(model.downs):
        # print(child.conv)
        for layer in child.conv:  # for each layer in the sequential child
            # print(f"layer: {layer}, type: {type(layer)}")
            if type(layer) == torch.nn.modules.conv.Conv2d:  # if the layer is Conv2d
                conv_layers.append(layer)
    print(f"conv_layers: {conv_layers}")

    # Retrieve the weights/filters of the conv layers
    conv_layers_weights = []
    for layer in conv_layers:
        conv_layers_weights.append(layer.weight)

    # Retrieve the featuremaps of the layers
    featuremaps = [conv_layers[0](image)]
    for x in range(1, len(conv_layers)):
        featuremaps.append(conv_layers[x](featuremaps[-1]))

    # Visualize the featuremaps
    for x in range(len(featuremaps)):
        plt.figure(figsize=(30, 30))
        plt.title(f"Feature maps of block {x}")
        layers = featuremaps[x][0, :, :, :].detach()
        for i, filter in enumerate(layers):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis('off')

        # plt.savefig('featuremap%s.png'%(x))
    plt.show()

    # Visualize the weights
    # ToDo: debug this:
    # Visualising the featuremaps
    # for x in range(len(featuremaps)):
    #     plt.figure(figsize=(30, 30))
    #     layers = featuremaps[x][0, :, :, :].detach()
    #     for i, filter in enumerate(layers):
    #         if i == 64:
    #             break
    #         plt.subplot(8, 8, i + 1)
    #         plt.imshow(filter, cmap='gray')
    #         plt.axis('off')
    #
    #     # plt.savefig('featuremap%s.png'%(x))
    #
    # plt.show()

    exit()


    # Process image to every layer and append output and name of the layer to outputs[] and names[] lists
    outputs = []
    names = []
    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    # convert 3D tensor to 2D, Sum the same element of every channel
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    # Plot feature maps and save them
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.show()
    # ToDo: specify the path to the image
    # plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
