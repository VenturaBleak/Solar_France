import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict

def calculate_classification_metrics(loader, model, probability_treshold, positive_pixel_threshold, device="cuda"):
    """Calculate common metrics for classification.

    Args:
        loader: DataLoader for the dataset
        model: Model to evaluate
        device: Device to use for evaluation
    Returns:
        A list of metrics
    """
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data, label, image_paths, image_dirs in loader:
            data = data.to(device)

            # Forward pass
            logits = model(data)

            # Turn logits into probabilities
            probs = torch.sigmoid(logits)

            # Turn probabilities into positive or negative predictions, depending on a certain threshold, e.g. 0.5
            preds = (probs > probability_treshold).float()

            # Turn predictions into a list of 0s and 1s
            total_pixels = preds.view(preds.size(0), -1).size(1)

            # Calculate the number of positive pixels in each image
            positive_pixel_count = torch.sum(preds.view(preds.size(0), -1), dim=1)

            # Determine whether each image is classified as positive or negative, depending on the threshold
            classification_preds = (positive_pixel_count / total_pixels > positive_pixel_threshold).cpu().numpy()

            # Extract true labels from the image paths
            true_labels.extend([1 if 'images_positive' in image_dir else 0 for image_dir in image_dirs])
            predicted_labels.extend(classification_preds)

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)

    model.train()

    return acc, f1, precision, recall, cm

class BinaryMetrics():
    """Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-8):
        # epsilon to avoid zero division error
        self.eps = eps

        # Initialize the self.metrics dictionary with None values
        self.metrics = {
            "val_loss": None,
            "balanced_acc": None,
            "pixel_acc": None,
            "iou": None,
            "f1_score": None,
            "precision": None,
            "recall": None,
            "specificity": None,
        }

    def __call__(self, y_true, y_pred):
        """Calculate overlap metrics.
        :param gt: ground truth
        :param pred: prediction
        """
        # Unit Test: asert y_pred.shape[1] == 1, 'Predictions must contain only one channel' \
        assert y_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                     ' when performing binary segmentation'

        # Flatten the tensors
        gt = y_true.to(y_pred.device, dtype=torch.float)
        pred = y_pred.view(-1, )
        target = gt.view(-1, ).float()

        # Calculate the confusion matrix
        tp = torch.sum(pred * target)  # TP
        fp = torch.sum(pred * (1 - target))  # FP
        fn = torch.sum((1 - pred) * target)  # FN
        tn = torch.sum((1 - pred) * (1 - target))  # TN

        #     Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)
        #     The proportion of correctly classified pixels (both foreground and background) out of the total number of pixels.
        #     Intuition: A higher pixel accuracy means better identification of both foreground and background classes.
        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)

        # IoU = TP / (TP + FP + FN)
        # The proportion of correctly classified foreground pixels out of the total number of pixels in both the ground truth and prediction.
        # Intuition: A higher IoU means better identification of the foreground class.
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)

        #     Specificity: TN / (TN + FP) -> also called true negative rate or background accuracy
        #     The proportion of true negative pixels (correctly classified background pixels) out of all TN and FP pixels.
        #     Intuition: A higher specificity means better identification of the background class.
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        #     Precision: TP / (TP + FP)
        #     The proportion of true positive pixels (correctly classified foreground pixels) out of all predicted positive pixels (both TP and FP).
        #     Intuition: A higher precision means fewer false positives.
        precision = (tp + self.eps) / (tp + fp + self.eps)

        #     Recall: TP / (TP + FN) -> also called foreground accuracy
        #     The proportion of true positive pixels (correctly classified foreground pixels) out of all actual positive pixels (both TP  and FN).
        #     Intuition: A higher recall means better identification of the foreground class.
        recall = (tp + self.eps) / (tp + fn + self.eps)

        #     F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        #     A measure of the balance between precision and recall.
        #     Intuition: A higher F1 score means better overlap between the predicted segmentation and the ground truth segmentation.
        f1_score = 2 * (precision * recall) / (precision + recall + self.eps)

        #   Balanced Accuracy: (Recall + Background Accuracy) / 2
        #   A measure of the balance between recall and background accuracy.
        #   Intuition: A higher balanced accuracy means better overlap between the predicted segmentation and the ground truth segmentation.
        #   Deals with the issue when there is only background in the ground truth segmentation.
        balanced_acc = (recall + specificity) / 2

        # store metrics in dict
        self.metrics = {
            "balanced_acc": balanced_acc,
            "pixel_acc": pixel_acc,
            "iou": iou,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }

        return self.metrics

    def calculate_binary_metrics(self, loader, model, loss_fn, device="cuda"):
        model.eval()

        total_metrics = defaultdict(float)
        epoch_loss = 0
        num_images = 0

        with torch.no_grad():
            for X, y, _, _ in loader:
                X = X.to(device)
                y = y.float().to(device)
                logits = model(X)
                loss = loss_fn(logits, y)
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()

                # calculate metrics
                metrics = self(y, preds)
                for metric_name, metric_value in metrics.items():
                    total_metrics[metric_name] += metric_value.item()

                # calculate epoch loss
                batch_size = X.size(0)
                epoch_loss += loss.item() * batch_size
                num_images += batch_size

        # Calculate average metrics
        num_batches = len(loader)
        avg_metrics = {name: (value / num_batches) for name, value in total_metrics.items()}

        # Calculate average loss per image
        avg_loss_per_image = epoch_loss / num_images
        avg_metrics["val_loss"] = avg_loss_per_image

        model.train()

        return avg_metrics