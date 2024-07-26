"""Telemetry module for Nirmata."""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from numpy import int_ as np_int
from sklearn.metrics import confusion_matrix


class Scorer:
    """A class for calculating evaluation scores and metrics for a classification model.

    :param y_true: The true labels.
    :type y_true: npt.NDArray
    :param y_pred: The predicted labels.
    :type y_pred: npt.NDArray
    :raises ValueError: If `y_true` or `y_pred` contains invalid labels.

    .. code-block:: python

        # Example usage with logging to MLflow:
        y_true = np.array(["full-compliance", "minor non-conformity"])
        y_pred = np.array(["full-compliance", "minor non-conformity"])

        # Score the results
        score = Scorer(y_true, y_pred)
        cm, cm_path, M_MAE, k = score.run_scores()

        # Log the scores
        mlflow.log_artifact(cm_path)
        mlflow.log_metric("Macro-averaged MAE", M_MAE)
        mlflow.log_metric("Kappa", k)

    """
    def __init__(self, y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> None:
        """Initialize a Telemetry object.

        This method initializes an object by setting the true and predicted labels.
        It also ensures that the labels are valid and consistent.

        :param y_true: The true labels.
        :type y_true: npt.NDArray
        :param y_pred: The predicted labels.
        :type y_pred: npt.NDArray
        :raises ValueError: If `y_true` or `y_pred` contains invalid labels.

        """
        #Define the valid labels for the classification
        self.labels = [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]

        # Ensure that y_ture and y_pred are not empty
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("y_true and y_pred must not be empty")

        # Check that y_true contains only the labels in the labels list
        if not all(label in self.labels for label in y_true):
            raise ValueError("y_true contains invalid labels")

        # Check that y_pred contains only the labels in the labels list
        if not all(label in self.labels for label in y_pred):
            raise ValueError("y_pred contains invalid labels")

        # Assign the true and predicted labels to the instance variables
        self.y_true = y_true
        self.y_pred = y_pred

        #Ensure that y_true and y_pred have the same length
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")

    def calculate_conf_matrix(self) -> tuple[npt.NDArray[np_int], str]:
        """Calculate the confusion matrix and save the corresponding plot.

        This method calculates the confusion matrix using the true and predicted labels,
        and then generates and saves a plot of the confusion matrix.

        :return: A tuple containing the confusion matrix and the path to the saved plot.
        :rtype: tuple[npt.NDArray[np_int], str]
        """
        cm: npt.NDArray[np_int] = confusion_matrix(
            self.y_true,                # Array of true labels
            self.y_pred,                # Array of predicted labels
            labels=self.labels,         # List of labels to index the matrix
        )

        # Plot and log the confusion matrix
        cm_path = self.__plot_and_log_conf_matrix(cm)

        return cm, cm_path

    def __plot_and_log_conf_matrix(self, cm: npt.NDArray[np_int]) -> str:
        """Plot and log the confusion matrix.

        This method generates a plot of the given confusion matrix
        using seaborn library and saves the plot to a specified path.

        :param cm: The confusion matrix.
        :type cm: npt.NDArray[np_int]
        :return: The path to the saved plot.
        :rtype: str

        """
        # Change figure size and increase dpi for better resolution
        plt.figure(figsize=(7, 5), dpi=100)

        # Scale up the size of all text in the plot
        sns.set_theme(font_scale=0.7)

        # Plot Confusion Matrix using Seaborn heatmap()
        ax = sns.heatmap(cm, annot=True, fmt="d")

        # set x-axis label and ticks.
        ax.set_xlabel("Predicted", fontsize=10, labelpad=10)
        ax.xaxis.set_ticklabels(self.labels)

        # set y-axis label and ticks
        ax.set_ylabel("Actual", fontsize=10, labelpad=10)
        ax.yaxis.set_ticklabels(self.labels)

        # set plot title
        ax.set_title("Confusion Matrix", fontsize=10, pad=10)

        # Save the confusion matrix plot to the specified path
        cm_path = "/tmp/cm.png"
        plt.savefig(cm_path)

        #Close the plot to free up the memory
        plt.close()

        return cm_path

    def __transform_to_numerical(self, y: npt.NDArray[np_int]) -> npt.NDArray[np_int]:
        """Transform labels to numerical values.

        Method converts labels to their corresponding label values.

        :param y: The labels to transform.
        :type y: npt.NDArray
        :return: The transformed labels.
        :rtype: npt.NDArray[np_int]

        """
        # Create a dictionary to map labels to numerical values
        label_map = {label: i for i, label in enumerate(self.labels)}

        # Transform labels to numerical values
        return np.array([label_map[label] for label in y])

    def calculate_M_MAE(self) -> float:
        """Calculate the macroaveraged MAE (Mean Absolute Error).

        Method calculates the macroaveraged MAE
        by transforming the categorical labels into numerical values,
        and then computing the weighted average of MAE for each class.
        The weights are determined by the inverse of the number of samples,
        ensuring that each class contrbiutes equally regardless of its size.

        Reference:
        - Baccianella et al. (2009) for calculating the metric.

        :return: The macroaveraged MAE.
        :rtype: float
        """
        # Trasform y_true and y_pred to numerical values in np array
        y_true_num = self.__transform_to_numerical(self.y_true)
        y_pred_num = self.__transform_to_numerical(self.y_pred)

        # Calculate macroaveraged MAE (look at Baccianella et al. 2009)
        macroMAE = 0.0

        # Calculate the MAE for each class and weight it
        for label in self.labels:
            # Find indices where the true label matches the current label
            label_indices = self.y_true == label

            # If the class is not present in the true labels, skip it
            if y_true_num[label_indices].size == 0:
                continue

            # Calculate the weight for the current class
            class_weight = 1.0 / y_true_num[label_indices].size

            # Calculate the MAE for the current class and it to macroMAE
            macroMAE += class_weight * np.sum(
                np.abs(y_true_num[label_indices] - y_pred_num[label_indices])
            )

        # Normalize the macroaveraged MAE by the number of label
        normalization_factor: float = 1.0 / len(self.labels)

        return macroMAE * normalization_factor

    def calculate_Cohen_Kappa(self) -> float:
        """Calculate Cohen's Kappa with ordinal data.

        This method computes Cohen's Kappa,
        a statistical measure of inter-rater agreement for categorical items,
        adjusted for agreement occurring by chance.
        It is suitable for ordinal data and uses quadratic weights to account for
        the degree of disagreement between different categories.

        :return: Cohen's Kappa.
        :rtype: float
        """
        # Trasform y_true and y_pred to numerical values in np array
        y_true_num = self.__transform_to_numerical(self.y_true)
        y_pred_num = self.__transform_to_numerical(self.y_pred)

        # Build the confusion matrix
        num_classes = len(self.labels)
        observed = confusion_matrix(
            y_true_num,
            y_pred_num,
            labels=list(range(num_classes))
        )
        num_sizes = float(len(y_true_num))

        # Build kappa weights matrix using quadratic weights
        kappa_weights = np.empty((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                kappa_weights[i, j] = (i - j)**2

        # Normalize the histogram of true and predicted labels
        hist_true = (
            np.bincount(y_true_num, minlength=num_classes)[:num_classes] / num_sizes
        )
        hist_pred = (
            np.bincount(y_pred_num, minlength=num_classes)[:num_classes] / num_sizes
        )

        # Compute the expected confusion matrix
        expected = np.outer(hist_true, hist_pred)

        # Normalize the observed confusion matrix
        observed = observed / num_sizes

        # Calculate Cohen's Kappa
        kappa: float = 1.0

        # If all weights are zero, that means no disagreements matter.
        if np.count_nonzero(kappa_weights):
            observed_sum = np.sum(kappa_weights * observed)
            expected_sum = np.sum(kappa_weights * expected)
            if observed_sum == 0.0:
                kappa = 1.0
            else:
                kappa -= float(np.sum(observed_sum) / np.sum(expected_sum))

        return kappa

    def run_scores(self) -> tuple[npt.NDArray[np_int], str, float, float]:
        """Run the scoring process and calculate evaluation scores.

        This method executes the entire scoring process,
        including the calculation of the confusion matrix,
        the macroaveraged Mean Absolute Error (MAE), and Cohen's Kappa.
        It returns the aforementioned evaluation metrics
        along with the path to the saved confusion matrix plot.

        :return: A tuple containing:
            - The confusion matrix as a NumPy array.
            - The path to the saved confusion matrix plot as a string.
            - The macroaveraged MAE as a float.
            - Cohen's Kappa as a float.
        :rtype: tuple[npt.NDArray[np_int], str, float, float]
        """
        # Calculate the confusion matrix and log it
        cm, cm_path = self.calculate_conf_matrix()

        # Calculate macroaveraged MAE
        M_MAE = self.calculate_M_MAE()

        # Calculate Cohen's Kappa with ordinal data
        kappa = self.calculate_Cohen_Kappa()

        return cm, cm_path, M_MAE, kappa
