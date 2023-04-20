import torch
import torchvision
from dataset import FranceSegmentationDataset
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F

def generate_model_name(architecture, loss_function, optimizer, dataset_names):
    model_name = f"{architecture}_{loss_function}_{optimizer}"
    for dataset_name in dataset_names:
        model_name += f"_{dataset_name}"
    return model_name

def save_checkpoint(state, filename="my_checkpoint.pth.tar", model_name=None, parent_dir=None):
    if model_name is not None:
        os.makedirs(os.path.join(parent_dir, "trained_models"), exist_ok=True)
        filepath = os.path.join(parent_dir, "trained_models")
        filename = os.path.join(parent_dir, "trained_models", f"{model_name}.pth.tar")

    torch.save(state, filename)
    print(f'Best model saved as {filename}')
    return filepath

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    image_dirs,
    mask_dirs,
    train_images,
    train_masks,
    val_images,
    val_masks,
    batch_size,
    train_transforms,
    val_transforms,
    num_workers=0,
    pin_memory=True,
):
    train_ds = FranceSegmentationDataset(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        images=train_images,
        masks=train_masks,
        transform=train_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = FranceSegmentationDataset(
        image_dirs=image_dirs,
        mask_dirs=mask_dirs,
        images=val_images,
        masks=val_masks,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def save_predictions_as_imgs(loader, model, unnorm, model_name, folder="saved_images/", device="cuda", testing=False):
    if testing == True:
        name_extension = "test"
    else:
        name_extension = "val"

    # create a folder if not exists, cwd + folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    # set model to eval mode
    model.eval()
    all_images = []
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Move x, y, and preds back to the CPU
            x = x.cpu()
            y = y.cpu()
            preds = preds.cpu()

            # Unnormalize the input image
            for i in range(x.size(0)):
                x[i] = unnorm(x[i])

            # Normalize the input image back to the range [0, 1]
            x = (x - x.min()) / (x.max() - x.min())

            # Repeat the single channel of y and preds 3 times to match the number of channels in x
            y = y.repeat(1, 3, 1, 1)
            preds = preds.repeat(1, 3, 1, 1)

            # Concatenate the image, ground truth mask, and prediction along the width dimension (dim=3)
            combined = torch.cat((x, y, preds), dim=3)
            all_images.append(combined)

            # Break after the third batch
            if idx == 2:
                break

    # Stack all images vertically
    stacked_images = torch.cat(all_images, dim=2)

    # specify the path to save the stacked images
    path = os.path.join(folder, f"{model_name}_{name_extension}.png")
    # Save the stacked images
    torchvision.utils.save_image(stacked_images, path)

    # Set the model back to train mode
    model.train()

def calculate_binary_metrics(loader, model, loss_fn, device="cuda"):
    """Calculate common metrics in binary cases.
    binary metrics: pixel accuracy, iou, precision, specificity, recall
    pixel accuracy = (TP + TN) / (TP + TN + FP + FN) -> intuition: how many pixels are correctly classified
    iou = TP / (TP + FP + FN) -> intuition: how many pixels are correctly classified as foreground
    precision = TP / (TP + FP) -> intuition:
    specificity = TN / (TN + FP) -> intuition:
    recall = TP / (TP + FN) -> intuition:

    Inspired by the following GitHub repository:
    Link: https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch/blob/master/metric.py

    Args:
        :param loader: DataLoader for the dataset
        :param model: Model to evaluate
        :param device: Device to use for evaluation
    Returns:
        :return: A list of metrics
        """
    metrics_calculator = BinaryMetrics()
    model.eval()

    total_metrics = [0, 0, 0, 0, 0, 0, 0]

    epoch_loss = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.float().to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            epoch_loss += loss.item()
            metrics = metrics_calculator(y, preds)
            total_metrics = [m + n for m, n in zip(total_metrics, metrics)]

    # calculate average metrics
    num_batches = len(loader)
    avg_metrics = [metric / num_batches for metric in total_metrics]

    # calculate average epoch loss
    epoch_loss = epoch_loss / num_batches
    avg_metrics.append(epoch_loss)

    model.train()

    return avg_metrics

class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative
    rate, as specificity/TPR is meaningless in multiclass cases.
    """

    def __init__(self, eps=1e-5):
        # epsilon to avoid zero division error
        self.eps = eps

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        #     Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)
        #     The proportion of correctly classified pixels (both foreground and background) out of the total number of pixels.
        #     Intuition: A higher pixel accuracy means better identification of both foreground and background classes.
        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)

        # IoU = TP / (TP + FP + FN)
        # The proportion of correctly classified foreground pixels out of the total number of pixels in both the ground truth and prediction.
        # Intuition: A higher IoU means better identification of the foreground class.
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)

        #     Specificity: TN / (TN + FP)
        #     The proportion of true negative pixels (correctly classified background pixels) out of all TN and FP pixels.
        #     Intuition: A higher specificity means better identification of the background class.
        precision = (tp + self.eps) / (tp + fp + self.eps)

        #     Precision: TP / (TP + FP)
        #     The proportion of true positive pixels (correctly classified foreground pixels) out of all predicted positive pixels (both TP and FP).
        #     Intuition: A higher precision means fewer false positives.
        recall = (tp + self.eps) / (tp + fn + self.eps)

        #     Recall: TP / (TP + FN) -> also called foreground accuracy
        #     The proportion of true positive pixels (correctly classified foreground pixels) out of all actual positive pixels (both TP  and FN).
        #     Intuition: A higher recall means better identification of the foreground class.
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        #     F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        #     A measure of the balance between precision and recall.
        #     Intuition: A higher F1 score means better overlap between the predicted segmentation and the ground truth segmentation.
        f1_score = 2 * (precision * recall) / (precision + recall + self.eps)

        #     Background Accuracy: TN / (TN + FP)
        #     The proportion of true negative pixels (correctly classified background pixels) out of all TN and FP pixels.
        #     Intuition: A higher specificity means better identification of the background class.
        bg_acc = tn / (tn + fp + self.eps)

        # convert to numpy & put on device -> cpu
        pixel_acc = pixel_acc.cpu().numpy()
        iou = iou.cpu().numpy()
        precision = precision.cpu().numpy()
        specificity = specificity.cpu().numpy()
        recall = recall.cpu().numpy()
        f1_score = f1_score.cpu().numpy()
        bg_acc = bg_acc.cpu().numpy()

        return pixel_acc, iou, precision, specificity, recall, f1_score, bg_acc

    def __call__(self, y_true, y_pred):
        assert y_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                     ' when performing binary segmentation'
        pixel_acc, iou, precision, specificity, recall, f1_score, bg_acc = self._calculate_overlap_metrics(
            y_true.to(y_pred.device,
                      dtype=torch.float),
            y_pred)
        return [pixel_acc, iou, precision, specificity, recall, f1_score, bg_acc]