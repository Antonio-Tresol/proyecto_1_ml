# Plots
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
from typing import List
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_general_metrics_barplot(general_metrics: pd.DataFrame):
    """
    Creates a bar plot, where each bar is a metric, the y axis is the value and the x axis is the model.

    Args:
        general_metrics (pd.DataFrame): General metrics table.
    """
    general_metrics_melt = general_metrics.reset_index().melt("index")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="index",
        y="value",
        hue="variable",
        data=general_metrics_melt,
        linewidth=1,
        edgecolor=".5",
    )
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.title("General Metrics")
    plt.show()


def plot_general_metrics_lineplot(general_metrics: pd.DataFrame):
    """
    Creates a line plot, where each line is a metric, the y axis is the value and the x axis is the model.

    Args:
        general_metrics (pd.DataFrame): General metrics table.
    """
    plt.figure(figsize=(8, 4))
    general_metrics.T.plot(kind="line", marker="o")
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.title("General Metrics")
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: List[List[float]],
    labels: List[str] = None,
    title: str = None,
    colors: object = None,
):
    """
    Plots a confusion matrix.

    Args:
        confusion_matrix (List[List[float]]): Confusion matrix.
        labels (List[str], optional): Display labels. Defaults to None (number from 0 to n_classes are chosen).
        title (str, optional): Title for the plot. Defaults to None.
        colors (object, optional): Colors used for the heat map. Defaults to None.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=labels
    )
    disp.plot(cmap=colors)

    # Display title
    if title is not None:
        disp.ax_.set_title(title)

    plt.show()


def plot_metrics(
    metrics: pd.DataFrame,
    algorithm: str,
    title: str = None,
    color: object = None,
):
    """
    Plot metrics as a bar plot.

    Args:
        metrics (pd.DataFrame): Pandas DataFrame with metrics.
        algorithm (str): Name of the algorithm used to getmetrics.
        title (str, optional): Title for the plot.
        colors (object, optional): Colors used for the heat map. Defaults to None.
    """
    """
    # Create the bar plot
    """
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    plt.figure(figsize=(8, 4))
    if color is not None:
        bar_plot = plt.barh(
            metrics["Metric"],
            metrics[algorithm],
            color=color(rescale(metrics[algorithm])),
        )
    else:
        bar_plot = plt.barh(metrics["Metric"], metrics[algorithm])

    # Add a border
    for bar in bar_plot:
        bar.set_edgecolor("black")
        bar.set_linewidth(0.5)

    x_offset = 0.01
    # Add exact values on the bars
    for index, value in enumerate(metrics[algorithm]):
        plt.text(
            value + x_offset, index, f"{value:.4f}", va="center", ha="left", fontsize=8
        )

    # Add labels and title
    plt.xlabel("Value")
    plt.ylabel("Metric")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Model Evaluation Metrics")

    # Increase limits
    plt.xlim(0, max(metrics[algorithm]) * 1.15)

    # Invert y-axis to have the bars ordered top to bottom
    plt.gca().invert_yaxis()
    plt.show()
