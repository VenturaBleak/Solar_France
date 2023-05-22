import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        return (model_output[0, :, :] * self.mask).sum()

def visualize_gradcam_UNET(model, val_loader, file_name, folder, device):
    """
    Visualize the gradient class activation maps of the model.

    Parameters:
    model: PyTorch model whose gradcam are to be visualized.
    val_loader: Validation data loader.
    file_name: Name of the file to save the gradcam visualizations to.
    device: Device to perform computations on.
    """
    SHOW = False

    device = torch.device(device)

    original_model_mode = model.training
    model.eval()

    target_layers = [model.downs[3].conv[3]]

    X_val, y_val = next(iter(val_loader))
    X_val = X_val.to(device)

    with torch.no_grad():
        logits = model(X_val)
        preds = torch.sigmoid(logits)  # convert logits to probabilities

    # specify the target category and prediction threshold
    target_category = 0
    threshold = preds.max() * 0.95
    target_mask = (preds[0, target_category, :, :] > threshold).float()

    targets = [SemanticSegmentationTarget(target_category, target_mask)]

    # specify the CAM visualization methods
    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda'))]

    num_images = 8

    for method_name, cam in methods:
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))  # Setup a 2x4 grid of subplots
        with cam as c:
            for i in range(num_images):
                grayscale_cam = c(input_tensor=X_val[i].unsqueeze(0), targets=targets)[0, :]
                img = X_val[i].cpu().numpy().transpose(1, 2, 0)
                img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)

                cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                row = i // 4  # Change here for a 2x4 grid
                col = i % 4  # Change here for a 2x4 grid
                axs[row, col].imshow(cam_image)
                axs[row, col].set_title(f'Image {i + 1}', fontsize=18)
                axs[row, col].axis('off')

            # Add a title to the figure
            fig.suptitle(method_name, fontsize=28)

            # show the image
            if SHOW == True:
                fig.show()

            # Save the figure as an image file
            image_file_path = os.path.join(folder, f"{file_name}_{method_name}.png")
            fig.savefig(image_file_path, bbox_inches='tight', pad_inches=0)

    model.train(mode=original_model_mode)

def visualize_gradcam_Segformer(model, val_loader, device):
    """
    Visualize the gradient class activation maps of the model.

    Parameters:
    model: PyTorch model whose gradcam are to be visualized.
    val_loader: Validation data loader.
    device: Device to perform computations on.
    """
    device = torch.device(device)

    original_model_mode = model.training
    model.eval()

    # Assuming that we want to apply Grad-CAM on the output of last self-attention layer of each stage.
    # The target_layers will be a list of last self-attention layers of each stage.
    print(model.mit.stages)
    print(model.mit.stages[3][2][1][1])
    print(model.mit.stages[3][2][1][1].fn.net[3])

    # test
    sample_input = torch.rand(1, 3, 416, 416)
    intermediate_output = model.mit.stages[0](sample_input)
    intermediate_output = model.mit.stages[1](intermediate_output)
    intermediate_output = model.mit.stages[2](intermediate_output)
    intermediate_output = model.mit.stages[3](intermediate_output)
    print(intermediate_output.size())
    exit()