# Import necessary modules
import numpy as np

# Data handling
import pandas as pd

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# Model selection tools
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Utils
from typing import Tuple, List


def make_grid_search(
    model: object, search_space: dict, X: pd.DataFrame, Y: pd.Series, verbose: int = 1
) -> tuple:
    """
    Perform grid search to find the best hyperparameters for a given model.

    Args:
        model (object): The machine learning model from the sci kit learn library to be used for grid search.
        search_space (dict): The hyperparameter search space.
        X (pd.DataFrame): The input features.
        Y (pd.Series): The target variable.
        verbose (int, optional): Verbosity level. Default is 1.

    Returns:
        tuple: A tuple containing the best score, best parameters, and the results of the grid search.

    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=search_space,
        scoring=["accuracy", "precision", "recall", "roc_auc"],
        refit="roc_auc",
        cv=5,
        verbose=verbose,
    )

    grid_search.fit(X, Y)
    results = pd.DataFrame(grid_search.cv_results_)
    return grid_search.best_score_, grid_search.best_params_, results


def fit_and_get_metrics(
    model: object, algorithm: str, X: pd.DataFrame, Y: pd.Series, n_folds: int
) -> Tuple[pd.DataFrame, List[List[float]]]:
    """
    Performs cross validation on a model and gets the metrics mean.

    Args:
        model (object): The machine learning model from the sci kit learn library to be used for cross validation.
        algorithm (str): Name of the algorithm being used.
        X (pd.DataFrame): The input features.
        Y (pd.Series): The target variable.
        n_folds (int): Number of folds for cross validation.

    Returns:
        Tuple[pd.DataFrame, List[List[float]]]: A data frame containing the mean values of accuracies, precisions, recalls, roc_auc_scores, and the confusion matrix.

    """
    # Get cross validation indices
    kf = StratifiedKFold(n_splits=n_folds)

    # Metrics
    accuracies = []
    precisions = []
    recalls = []
    rocs = []
    confusion = []

    # Cross validate
    for iteration, (train_index, test_index) in enumerate(kf.split(X, Y)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)

        # Get metrics
        y_predicted = model.predict(X_test)
        accuracies.append(accuracy_score(Y_test, y_predicted))
        precisions.append(precision_score(Y_test, y_predicted))
        recalls.append(recall_score(Y_test, y_predicted))
        rocs.append(roc_auc_score(Y_test, y_predicted))
        if iteration == 0:
            confusion = confusion_matrix(Y_test, y_predicted)

    columns = ["Accuracy", "Precision", "Recall", "Roc & Auc"]
    metrics = [
        np.mean(accuracies),
        np.mean(precisions),
        np.mean(recalls),
        np.mean(rocs),
    ]

    dataframe = {"Metric": columns, algorithm: metrics}
    return pd.DataFrame(dataframe), confusion


def get_general_metrics(algorithms_metrics: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Creates a panda dataframe that contains a table with the information of each metric for each algorithm.
    Columns are the metrics and rows are the algorithms.

    Args:
        algorithms_metrics (List[pd.DataFrame]): A list of data frames containing the metrics for an algorithm.

    Returns:
        pd.Dataframe: The table with the information of each metric for each algorithm.
    """
    for index, algorithm_metrics in enumerate(algorithms_metrics):
        algorithms_metrics[index] = algorithm_metrics.set_index("Metric").T
        # Remove the "Metric" column name
        algorithms_metrics[index].columns.name = None

    return pd.concat(algorithms_metrics).T
