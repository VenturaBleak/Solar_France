import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        return (model_output[0, :, :] * self.mask).sum()

def visualize_gradcam_UNET(model, val_loader, device):
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

    target_layers = [model.downs[3].conv[3]]

    X_val, y_val = next(iter(val_loader))
    X_val = X_val.to(device)

    with torch.no_grad():
        logits = model(X_val)
        preds = torch.sigmoid(logits)  # convert logits to probabilities

    target_category = 0
    threshold = preds.max() * 0.95
    target_mask = (preds[0, target_category, :, :] > threshold).float()

    targets = [SemanticSegmentationTarget(target_category, target_mask)]

    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')),
               ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda'))]

    for method_name, cam in methods:
        with cam as c:
            grayscale_cam = c(input_tensor=X_val, targets=targets)[0, :]
            img = X_val[0].cpu().numpy().transpose(1, 2, 0)
            img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)

            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            plt.imshow(cam_image)
            plt.title(method_name)
            plt.axis('off')
            plt.show()

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