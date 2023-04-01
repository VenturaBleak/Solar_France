import torch
from torch import nn
import torchvision
from dataset import FranceSegmentationDataset
from torch.utils.data import DataLoader
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    image_dir,
    mask_dir,
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
        image_dir=image_dir,
        mask_dir=mask_dir,
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
        image_dir=image_dir,
        mask_dir=mask_dir,
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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    class_correct = [0, 0]  # [background_correct, foreground_correct]
    class_pixels = [0, 0]  # [background_pixels, foreground_pixels]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum().item()) / (
                (preds + y).sum().item() + 1e-8
            )  # Convert to Python int

            # Overall accuracy calculation
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            # Class-wise accuracy calculation
            for cls in range(2):
                class_correct[cls] += ((preds == y) * (y == cls).float()).sum().item()
                class_pixels[cls] += (y == cls).sum().item()

    print(f"Dice score: {dice_score/len(loader):.3f}")
    print(f"Overall Accuracy: {num_correct/num_pixels*100:.2f}")
    print(f"Background class accuracy: {class_correct[0]/class_pixels[0]*100:.2f}")
    print(f"Foreground class accuracy: {class_correct[1]/class_pixels[1]*100:.2f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", epoch=0):
    # create a folder if not exists, cwd + folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    # set model to eval mode
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Move x, y, and preds back to the CPU
        x = x.cpu()
        y = y.cpu()
        preds = preds.cpu()

        # Normalize the input image back to the range [0, 1]
        x = (x - x.min()) / (x.max() - x.min())

        # Repeat the single channel of y and preds 3 times to match the number of channels in x
        y = y.repeat(1, 3, 1, 1)
        preds = preds.repeat(1, 3, 1, 1)

        # Concatenate the image, ground truth mask, and prediction along the width dimension (dim=3)
        combined = torch.cat((x, y, preds), dim=3)

        # Save the combined image
        torchvision.utils.save_image(combined, f"{folder}/Epoch_{epoch}.png")

        # Break after the first batch
        if idx == 0:
            break

    # Set the model back to train mode
    model.train()

def calculate_binary_metrics(loader, model, device="cuda"):
    """Calculate common metrics in binary cases.
    binary metrics: pixel accuracy, dice, precision, specificity, recall
    pixel accuracy = (TP + TN) / (TP + TN + FP + FN) -> intuition: how many pixels are correctly classified
    dice = 2 * (TP) / (2 * TP + FP + FN) -> intuition:
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

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            metrics = metrics_calculator(y, preds)
            total_metrics = [m + n for m, n in zip(total_metrics, metrics)]

    num_batches = len(loader)
    avg_metrics = [metric / num_batches for metric in total_metrics]

    pixel_acc, dice, precision, specificity, recall, f1_score, bg_acc = avg_metrics

    #print(f"Pixel Accuracy: {pixel_acc:.3f} | Dice Score: {dice:.3f} | Precision: {precision:.3f} | Specificity: {specificity:.3f} | Recall: {recall:.3f} | F1 Score: {f1_score:.3f} | Background Accuracy: {bg_acc:.3f}")
    # print only F1, precision, recall
    print(f"F1-Score:{f1_score:.3f} | Recall:{recall:.3f} | Precision:{precision:.3f}")

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

        #     Dice Score: 2 * (TP) / (2 * TP + FP + FN) -> Jaccard index
        #     A measure of similarity between the predicted segmentation and the ground truth segmentation, considering both TP and FP + FN pixels.
        #     Intuition: A higher dice score means better overlap between the predicted segmentation and the ground truth segmentation.
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)

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
        bg_acc = tn / (tn + fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall, f1_score, bg_acc

    def __call__(self, y_true, y_pred):
        assert y_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                     ' when performing binary segmentation'
        pixel_acc, dice, precision, specificity, recall, f1_score, bg_acc = self._calculate_overlap_metrics(
            y_true.to(y_pred.device,
                      dtype=torch.float),
            y_pred)
        return [pixel_acc, dice, precision, specificity, recall, f1_score, bg_acc]