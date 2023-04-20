import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_loss_and_lr(df):
    fig, ax1 = plt.subplots()

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color=plt.cm.viridis(0.1))
    ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", color=plt.cm.viridis(0.8))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["learning_rate"], label="Learning Rate", color=plt.cm.viridis(0.5), alpha=0.3)
    ax2.set_ylabel("Learning Rate")
    ax2.legend(loc="upper right")

    plt.title("Loss and Learning Rate vs Epoch")
    plt.show()


def plot_metric(df, metric):
    plt.plot(df["epoch"], df[metric], label=metric, color=plt.cm.viridis(0.6))
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Epoch")
    plt.legend()
    plt.show()


def plot_precision_recall_f1(df):
    plt.plot(df["epoch"], df["val_precision"], label="Precision", color=plt.cm.viridis(0.1))
    plt.plot(df["epoch"], df["val_recall"], label="Recall", color=plt.cm.viridis(0.5))
    plt.plot(df["epoch"], df["val_f1-score"], label="F1-score", color=plt.cm.viridis(0.9))
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Precision, Recall and F1-Score vs Epoch")
    plt.legend()
    plt.show()


def plot_acc_f1_iou(df):
    plt.plot(df["epoch"], df["val_pixel_acc"], label="Pixel Accuracy", color=plt.cm.viridis(0.1))
    plt.plot(df["epoch"], df["val_f1-score"], label="F1-score", color=plt.cm.viridis(0.5))
    plt.plot(df["epoch"], df["val_iou"], label="IoU", color=plt.cm.viridis(0.9))
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Pixel Accuracy, F1-score and IoU vs Epoch")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # Define the path to the trained models directory
    file_path = os.path.join(parent_dir, "trained_models")

    # specify the model name to evaluate
    MODEL_NAME = "B0_TverskyLoss_AdamW_France_google_Munich_Denmark_Heerlen_2018_HR_output_ZL_2018_HR_output_logs"

    # Load the DataFrame from CSV file
    df = pd.read_csv(os.path.join(file_path, f"{MODEL_NAME}.csv"))

    # Call the functions to generate plots
    plot_loss_and_lr(df)
    plot_metric(df, "val_f1-score") # replace with any other metric
    plot_precision_recall_f1(df)
    plot_acc_f1_iou(df)