import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
from matplotlib.lines import Line2D

def load_logs_in_folder(folder):
    """
    Load all log files in a specified folder into a dictionary of pandas DataFrames.
    Only the files with names ending with 'logs.csv' will be loaded.
    """
    log_files = glob.glob(os.path.join(folder, '*logs.csv'))
    logs = {}
    for log_file in log_files:
        # Remove the '_logs.csv' from the model name
        model_name = os.path.basename(log_file).replace('_logs.csv', '')
        logs[model_name] = pd.read_csv(log_file)
    return logs

def plot_measure(logs, measure, dir_path=None):
    """
    Plot a specified measure for multiple models
    """
    colors = cm.Accent(np.linspace(0, 1, len(logs)))
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10,8))
    for (model, df), color in zip(logs.items(), colors):
        if measure in df.columns:
            plt.plot(df["epoch"], df[measure], label=model, color=color)
        else:
            print(f"Column '{measure}' does not exist in log of model '{model}'")
    plt.xlabel("Epoch")
    plt.ylabel(measure)
    plt.title(f"{measure}")
    plt.legend()
    plt.show()

    # save the plot
    if dir_path is not None:
        plt.savefig(os.path.join(dir_path, f"{measure}_vs_epoch.png"))

def plot_loss_and_lr(df):
    fig, ax1 = plt.subplots()

    # give the plot a small padding on the right side
    fig.subplots_adjust(right=0.85)

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color=plt.cm.Accent(0.1))
    ax1.plot(df["epoch"], df["val_loss"], label="Test Loss", color=plt.cm.Accent(0.8))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["learning_rate"], label="Learning Rate", color=plt.cm.Accent(0.5), alpha=0.3)
    ax2.set_ylabel("Learning Rate")
    ax2.legend(loc="upper right")

    plt.title("Loss and Learning Rate")
    plt.show()


def plot_metric(df, metric):
    plt.plot(df["epoch"], df[metric], label=metric, color=plt.cm.Accent(0.6))
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric}")
    plt.legend()
    plt.show()


def plot_precision_recall_f1(df):
    plt.plot(df["epoch"], df["precision"], label="Precision", color=plt.cm.Accent(0.1))
    plt.plot(df["epoch"], df["recall"], label="Recall", color=plt.cm.Accent(0.5))
    plt.plot(df["epoch"], df["f1_score"], label="F1-score", color=plt.cm.Accent(0.9))
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Precision, Recall and F1-Score")
    plt.legend()
    plt.show()


def plot_acc_f1_iou(df):
    plt.plot(df["epoch"], df["pixel_acc"], label="Pixel Accuracy", color=plt.cm.Accent(0.1))
    plt.plot(df["epoch"], df["f1_score"], label="F1-score", color=plt.cm.Accent(0.5))
    plt.plot(df["epoch"], df["iou"], label="IoU", color=plt.cm.Accent(0.9))
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Pixel Accuracy, F1-score and IoU")
    plt.legend()
    plt.show()

def plot_correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Matrix of Metrics")
    plt.show()

def summarize_model_performance(df):
    metrics = ["train_loss", "val_loss", "balanced_acc", "pixel_acc", "iou", "f1_score", "precision", "recall",
               "specificity"]
    summary = pd.DataFrame(index=metrics)
    summary["min"] = df[metrics].min()
    summary["max"] = df[metrics].max()
    summary["final"] = df[metrics].iloc[-1]
    print(summary)

def summarize_models_performances(logs, dir_path):
    metrics = ["epoch", "train_loss", "val_loss", "balanced_acc", "pixel_acc", "iou", "f1_score", "precision", "recall",
               "specificity"]
    summaries = []
    for model, df in logs.items():
        max_f1_score_row = df[df['f1_score'] == df['f1_score'].max()].iloc[0]
        summary = {metric: max_f1_score_row[metric] for metric in metrics}
        summary['model'] = model
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    # reordering columns so 'model' is the first column
    cols = summary_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    summary_df = summary_df[cols]
    # show all columns, pandas
    pd.set_option('display.max_columns', None)
    print(summary_df)

    # save to csv
    summary_df.to_csv(os.path.join(dir_path, "summary.csv"), index=False)

def plot_all_measures(logs, measures, dir_path):
    """
    Plot specified measures for multiple models
    """
    colors = cm.Accent(np.linspace(0, 1, len(logs)))
    color_dict = {model:color for (model, _), color in zip(logs.items(), colors)}

    # make grid flexible to adjust to number of measures
    plots_x = int(math.ceil(math.sqrt(len(measures)+1)))
    plots_y = plots_x

    fig, axs = plt.subplots(plots_x, plots_y, figsize=(20,16))
    axs = axs.ravel()

    for ax, measure in zip(axs, measures):
        for model, df in logs.items():
            if measure in df.columns:
                ax.plot(df["epoch"], df[measure], label=model, color=color_dict[model])
            else:
                print(f"Column '{measure}' does not exist in log of model '{model}'")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(measure)
        ax.set_title(f"{measure.capitalize()}")

    # Remove axes for the last plot
    axs[-1].axis('off')

    # Creating a common legend in the last subplot
    custom_lines = [Line2D([0], [0], color=color, lw=4) for model, color in color_dict.items()]
    axs[-1].legend(custom_lines, logs.keys(), loc='center', fontsize='large')

    plt.tight_layout()
    plt.show()

    # save, high dpi
    fig.savefig(os.path.join(dir_path, "all_measures.png"), bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    # Get the current working directory
    cwd = os.getcwd()

    # Define the parent directory of the current working directory
    parent_dir = os.path.dirname(cwd)

    # Define the path to the trained models directory
    file_path = os.path.join(parent_dir, "models")
    experiment_dir = os.path.join(file_path, "Experiment2")

    ########################################
    # Plot single model
    ########################################
    # specify the model name to evaluate
    MODEL_NAME = "UNet_NL"

    # Load the DataFrame from CSV file
    df = pd.read_csv(os.path.join(experiment_dir, f"{MODEL_NAME}_logs.csv"))

    # Call the functions to generate plots
    plot_loss_and_lr(df)
    plot_metric(df, "f1_score")
    plot_precision_recall_f1(df)
    plot_acc_f1_iou(df)
    plot_correlation_matrix(df)
    summarize_model_performance(df)

    ########################################
    # Plot multiple models in one plot
    ########################################
    logs = load_logs_in_folder(experiment_dir)

    # summarize the performance of all models
    summarize_models_performances(logs, experiment_dir)


    # To plot test loss
    plot_measure(logs, "val_loss", experiment_dir)

    # To plot f1_score
    plot_measure(logs, "f1_score", experiment_dir)

    # To plot multiple measures
    plot_all_measures(logs,
                      ["val_loss", "iou", "f1_score", "recall", "precision", "specificity", "balanced_acc", "pixel_acc"],
                      experiment_dir)