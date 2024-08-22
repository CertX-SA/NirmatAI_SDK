# type: ignore
""""Unit tests for the telemetry module of NirmatAI."""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from nirmatai_sdk.telemetry import Scorer


def test_scorer_valid_labels():
    """Initialize without error if y_true and y_pred contain valid labels."""
    y_true = np.array(
        ["full-compliance", "minor non-conformity", "major non-conformity"]
    )
    y_pred = np.array(
        ["full-compliance", "minor non-conformity", "major non-conformity"]
    )

    scorer = Scorer(y_true, y_pred)
    assert np.array_equal(scorer.y_true, y_true)
    assert np.array_equal(scorer.y_pred, y_pred)


def test_scorer_invalid_y_true_labels():
    """Raise a ValueError if y_true contains invalid labels."""
    y_true = np.array(["C", "B", "B"])
    y_pred = np.array(
        ["full-compliance", "minor non-conformity", "major non-conformity"]
    )

    with pytest.raises(ValueError, match="y_true contains invalid labels"):
        Scorer(y_true, y_pred)


def test_scorer_invalid_y_pred_labels():
    """Raise a ValueError if y_pred contains invalid labels."""
    y_true = np.array(
        ["full-compliance", "minor non-conformity", "major non-conformity"]
    )
    y_pred = np.array(["C", "B", "B"])

    with pytest.raises(ValueError, match="y_pred contains invalid labels"):
        Scorer(y_true, y_pred)


def test_scorer_mismatched_lengths():
    """Raise a ValueError if y_true and y_pred have different lengths."""
    y_true = np.array(
        ["full-compliance", "minor non-conformity", "major non-conformity"]
    )
    y_pred = np.array(["full-compliance", "minor non-conformity"])

    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        Scorer(y_true, y_pred)


def test_scorer_empty_arrays():
    """Raise a ValueError if y_true or y_pred are empty."""
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError, match="y_true and y_pred must not be empty"):
        Scorer(y_true, y_pred)


def test_calculate_conf_matrix_cm():
    """When all labels are correct, the confusion matrix SHALL be a diagonal matrix.

    The elements in the main diagonal SHALL represent the number of correct labels.
    """
    y_true = np.array(
        [
            "full-compliance",
            "full-compliance",
            "full-compliance",
            "minor non-conformity",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "full-compliance",
            "full-compliance",
            "full-compliance",
            "minor non-conformity",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    cm, cm_path = scorer.calculate_conf_matrix()
    assert np.array_equal(cm, np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))
    assert cm_path == "/tmp/cm.png"


def test_calculate_conf_matrix_one_label():
    """On a label case, the confusion matrix SHALL be a nxn matrix, n is # labels.

    The confusion matrix plot path SHALL be returned in /tmp directory.
    """
    y_true = np.array(["full-compliance"])
    y_pred = np.array(["full-compliance"])

    scorer = Scorer(y_true, y_pred)

    cm, cm_path = scorer.calculate_conf_matrix()

    assert np.array_equal(cm, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert cm_path == "/tmp/cm.png"


def test__transform_to_numerical():
    """Scorer SHALL transform labels to numerical values [0, n-1]."""
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    assert np.array_equal(
        scorer._Scorer__transform_to_numerical(y_true), np.array([0, 1, 2])
    )
    assert np.array_equal(
        scorer._Scorer__transform_to_numerical(y_pred), np.array([0, 1, 2])
    )


def test_calculate_M_MAE_all_correct():
    """When all labels are correct, the M MAE SHALL be 0.0."""
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    assert scorer.calculate_M_MAE() == 0.0


def test_calculate_M_MAE_one_wrong_low():
    """When:.

    - There is no class imbalance
    - One label is misclassified with its successive class

    The M MAE SHALL be 1/3.
    """
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "minor non-conformity",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    assert scorer.calculate_M_MAE() == 0.3333333333333333


def test_calculate_M_MAE_one_wrong_high():
    """When:.

    - There is no class imbalance
    - One label is misclassified with its two successive classes

    The M MAE SHALL be 2/3.
    """
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "major non-conformity",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    assert scorer.calculate_M_MAE() == 0.6666666666666666


def test_calculate_M_MAE_all_wrong_low():
    """When:.

    - There is no class imbalance
    - All labels are misclassified with their successive classes

    The M MAE SHALL be 1.0.
    """
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "minor non-conformity",
            "major non-conformity",
            "minor non-conformity",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    assert scorer.calculate_M_MAE() == 1.0


def test_calculate_M_MAE_all_wrong_high():
    """When:.

    - There is no class imbalance
    - All labels are misclassified with their two successive classes

    The M MAE SHALL be 4/3
    """
    y_true = np.array(
        [
            "full-compliance",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "major non-conformity",
            "full-compliance",
        ]
    )
    scorer = Scorer(y_true, y_pred)
    assert scorer.calculate_M_MAE() == 1.3333333333333333


def test_calculate_M_MAE_class_imbalance_min_class_wrong():
    """When:.

    - There is class imbalance
    - The class with the lowest number of samples is misclassified

    The Macroaveraged MAE SHALL be higher than the mean MAE calculated using
    sklearn.metrics.mean_absolute_error.
    """
    y_true = np.array(
        [
            # Repeat full-compliance 100 times
            *["full-compliance"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["minor non-conformity"] * 1,
        ]
    )
    y_pred = np.array(
        [
            # Repeat full-compliance 100 times
            *["full-compliance"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["major non-conformity"] * 1,
        ]
    )
    scorer = Scorer(y_true, y_pred)
    # Caclulate the mean MAE using sklearn.metrics.mean_absolute_error
    y_pred = scorer._Scorer__transform_to_numerical(y_pred)
    y_true = scorer._Scorer__transform_to_numerical(y_true)
    MAE = mean_absolute_error(y_true, y_pred)

    assert scorer.calculate_M_MAE() > MAE


def test_calculate_M_MAE_class_imbalance_maj_class_wrong():
    """When:.

    - There is class imbalance
    - The class with the highest number of samples is misclassified

    The Macroaveraged MAE SHALL be lower than the mean MAE calculated using
    sklearn.metrics.mean_absolute_error.
    """
    y_true = np.array(
        [
            # Repeat full-compliance 100 times
            *["full-compliance"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["major non-conformity"] * 1,
        ]
    )
    y_pred = np.array(
        [
            # Repeat full-compliance 100 times
            *["major non-conformity"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["major non-conformity"] * 1,
        ]
    )
    scorer = Scorer(y_true, y_pred)
    # Caclulate the mean MAE using sklearn.metrics.mean_absolute_error
    y_pred = scorer._Scorer__transform_to_numerical(y_pred)
    y_true = scorer._Scorer__transform_to_numerical(y_true)
    MAE = mean_absolute_error(y_true, y_pred)

    assert scorer.calculate_M_MAE() < MAE


def test_calculate_M_MAE_class_imbalance_all_wrong():
    """When:.

    - There is class imbalance
    - All labels are misclassified

    The Macroaveraged MAE SHALL be 5/3 (worst case).
    """
    y_true = np.array(
        [
            # Repeat full-compliance 100 times
            *["full-compliance"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["major non-conformity"] * 1,
        ]
    )
    y_pred = np.array(
        [
            # Repeat full-compliance 100 times
            *["major non-conformity"] * 100,
            # Repeat minor non-conformity 10 times
            *["full-compliance"] * 10,
            # Repeat major non-conformity 1 time
            *["full-compliance"] * 1,
        ]
    )
    scorer = Scorer(y_true, y_pred)

    # Worst case will have 5/3
    assert scorer.calculate_M_MAE() == 1.6666666666666665


def test_calculate_M_MAE_naive_classifier():
    """The Naive classifier SHALL have a M MAE around 8/9."""
    # Draw a random sample from a uniform distribution of labels
    y_true = np.random.choice(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ],
        1000000,
    )
    # Do the same for the predicted labels
    y_pred = np.random.choice(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ],
        1000000,
    )

    scorer = Scorer(y_true, y_pred)
    # Calculate M MAE
    M_MAE = scorer.calculate_M_MAE()

    # In the limit the M_MAE should be 8/9
    assert M_MAE >= 8 / 9 - 0.01
    assert M_MAE <= 8 / 9 + 0.01


def test_calculate_Cohen_Kappa_ordinal_case():
    """Kappa SHALL be higher for errors with higher magnitudes."""
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    # Case in which the error is higher
    y_pred_case_high = np.array(
        [
            "major non-conformity",
            "minor non-conformity",
            "full-compliance",
        ]
    )
    # Case in which the error is lower
    y_pred_case_low = np.array(
        [
            "minor non-conformity",
            "minor non-conformity",
            "major non-conformity",
        ]
    )

    scorer = Scorer(y_pred=y_pred_case_high, y_true=y_true)
    result_high = scorer.calculate_Cohen_Kappa()

    scorer = Scorer(y_pred=y_pred_case_low, y_true=y_true)
    result_low = scorer.calculate_Cohen_Kappa()

    assert result_high < result_low


def test_calculate_Cohen_naive_classifier():
    """The Naive classifier SHALL have a Kappa around 0.0."""
    # Draw a random sample from a uniform distribution of labels
    y_true = np.random.choice(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ],
        1000000,
    )
    # Do the same for the predicted labels
    y_pred = np.random.choice(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ],
        1000000,
    )

    scorer = Scorer(y_true, y_pred)
    k = scorer.calculate_Cohen_Kappa()

    assert abs(k) <= 0.01


def test_calculate_Cohen_Kappa_worst_case():
    """The worst case for Kappa SHALL be -1.0."""
    y_true = np.array(
        [
            "full-compliance",
            "major non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "major non-conformity",
            "full-compliance",
        ]
    )

    scorer = Scorer(y_pred=y_pred, y_true=y_true)
    k = scorer.calculate_Cohen_Kappa()

    assert k == -1.0


def test_calculate_Cohen_Kappa_imbalanced_classes():
    """When:.

    - Classes are imbalanced

    Kappa SHALL be lower for the class with the highest number of samples.
    """
    y_true = np.array(
        [
            # Repeat full-compliance 100 times
            *["full-compliance"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["major non-conformity"] * 1,
        ]
    )

    y_pred_wrong_min = np.array(
        [
            # Repeat full-compliance 100 times
            *["full-compliance"] * 100,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["full-compliance"] * 1,
        ]
    )

    y_pred_wrong_maj = np.array(
        [
            "major non-conformity",
            # Repeat full-compliance 100 times
            *["full-compliance"] * 99,
            # Repeat minor non-conformity 10 times
            *["minor non-conformity"] * 10,
            # Repeat major non-conformity 1 time
            *["major non-conformity"] * 1,
        ]
    )

    scorer = Scorer(y_pred=y_pred_wrong_min, y_true=y_true)
    k_wrong_min = scorer.calculate_Cohen_Kappa()

    scorer = Scorer(y_pred=y_pred_wrong_maj, y_true=y_true)
    k_wrong_maj = scorer.calculate_Cohen_Kappa()

    assert k_wrong_min < k_wrong_maj


def test_calculate_Cohen_Kappa_wrt_non_ordinal_low():
    """Kappa SHALL be equal to cohen_kappa_score for errors with lower magnitudes."""
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "minor non-conformity",
            "minor non-conformity",
        ]
    )

    scorer = Scorer(y_pred=y_pred, y_true=y_true)
    k = scorer.calculate_Cohen_Kappa()
    k_sklearn = cohen_kappa_score(
        y_true,
        y_pred,
    )

    assert k == k_sklearn


def test_calculate_Cohen_Kappa_wrt_non_ordinal_high():
    """Kappa SHALL be less than cohen_kappa_score for errors with higher magnitudes."""
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
        ]
    )
    y_pred = np.array(
        [
            "major non-conformity",
            "minor non-conformity",
        ]
    )

    scorer = Scorer(y_pred=y_pred, y_true=y_true)
    k = scorer.calculate_Cohen_Kappa()
    k_sklearn = cohen_kappa_score(
        y_true,
        y_pred,
    )

    assert k < k_sklearn


def test_run_scores():
    """When run_scores() is called, it SHALL return a tuple of:.

    - A confusion matrix as a numpy array.
    - A path to the confusion matrix plot as a string.
    - A Macro-averaged MAE as a float.
    - A Cohen's Kappa as a float.
    """
    y_true = np.array(["full-compliance"])
    y_pred = np.array(["full-compliance"])

    scorer = Scorer(y_pred=y_pred, y_true=y_true)

    result = scorer.run_scores()

    # Assert that the function returns a tuple of npt.NDArray[np_int] and str
    assert isinstance(result, tuple)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], str)
    assert isinstance(result[2], float)
    assert isinstance(result[3], float)


def test_run_scores_correct_single_label():
    """When:.

    - A label is provided
    - The label is correct

    run_scores() SHALL return:
    - A confusion matrix which is diagonal.
    - A path to the confusion matrix plot in /tmp directory.
    - A Macro-averaged MAE of 0.0.
    - A Cohen's Kappa of 1.0.
    """
    y_true = np.array(["full-compliance"])
    y_pred = np.array(["full-compliance"])

    scorer = Scorer(y_pred=y_pred, y_true=y_true)

    result = scorer.run_scores()

    # Assert that the function returns a tuple of npt.NDArray[np_int] and str
    assert np.array_equal(
        result[0],
        np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert result[1], "/tmp/cm.png"
    assert result[2] == 0.0
    assert result[3] == 1.0


def test_run_scores_wrong_single_label():
    """When:.

    - A label is provided
    - The label is incorrect

    run_scores() SHALL return:
    - A confusion matrix which is not diagonal.
    - A path to the confusion matrix plot in /tmp directory.
    - A Macro-averaged MAE greater than 0.0.
    - A Cohen's Kappa less than 1.0.
    """
    y_true = np.array(["full-compliance"])
    y_pred = np.array(["major non-conformity"])

    scorer = Scorer(y_pred=y_pred, y_true=y_true)

    result = scorer.run_scores()

    # Assert that the function returns a tuple of npt.NDArray[np_int] and str
    assert np.array_equal(
        result[0],
        np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert result[1], "/tmp/cm.png"
    assert result[2] > 0.0
    assert result[3] < 1.0


def test_run_scores_correct():
    """When all labels are correct, run_scores() SHALL return:.

    - A confusion matrix with a diagonal matrix.
    - A path to the confusion matrix plot in /tmp directory.
    - A Macro-averaged MAE of 0.0.
    - A Cohen's Kappa of 1.0.
    """
    y_true = np.array(
        [
            "full-compliance",
            "minor non-conformity",
            "major non-conformity",
        ]
    )
    y_pred = y_true

    scorer = Scorer(y_pred=y_pred, y_true=y_true)

    result = scorer.run_scores()

    assert np.array_equal(
        result[0],
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert result[1], "/tmp/cm.png"
    assert result[2] == 0.0
    assert result[3] == 1.0


# More Unit Tests for ensuring the reliability
def test_scorer_large_arrays_perfect_prediction():
    """Ensure Scorer calculates correct metrics for large arrays."""
    y_true = np.array(["full-compliance"] * 1000000)
    y_pred = np.array(["full-compliance"] * 1000000)

    scorer = Scorer(y_true, y_pred)
    result = scorer.run_scores()

    expected_cm = np.array([[1000000, 0, 0], [0, 0, 0], [0, 0, 0]])

    assert np.array_equal(result[0], expected_cm)
    assert result[1] == "/tmp/cm.png"
    assert result[2] == 0.0
    assert result[3] == 1.0


def test_scorer_large_arrays_random_predictions():
    """Ensure Scorer calculates correct metrics for large arrays."""
    np.random.seed(42)
    labels = [
        "full-compliance",
        "minor non-conformity",
        "major non-conformity",
    ]
    y_true = np.random.choice(labels, 1000000)
    y_pred = np.random.choice(labels, 1000000)

    scorer = Scorer(y_true, y_pred)
    result = scorer.run_scores()

    assert result[1] == "/tmp/cm.png"
    assert 8 / 9 - 0.01 <= result[2] <= 8 / 9 + 0.01
    assert -0.01 <= result[3] <= 0.01


def test_scorer_large_arrays_one_class_misclassified():
    """Ensure Scorer calculates correct metrics for large arrays."""
    y_true = np.array(["full-compliance"] * 500000 + ["minor non-conformity"] * 500000)
    y_pred = np.array(["full-compliance"] * 500000 + ["major non-conformity"] * 500000)

    scorer = Scorer(y_true, y_pred)
    result = scorer.run_scores()

    expected_cm = np.array([[500000, 0, 0], [0, 0, 500000], [0, 0, 0]])

    assert np.array_equal(result[0], expected_cm)
    assert result[1] == "/tmp/cm.png"
    assert result[2] == 1.0 / 3
    assert result[3] == 0.6666666666666667


def test_scorer_large_arrays_one_incorrect_entry():
    """Ensure Scorer calculates correct metrics for large arrays."""
    y_true = np.array(["full-compliance"] * 999999 + ["minor non-conformity"] * 1)
    y_pred = np.array(["full-compliance"] * 999999 + ["major non-conformity"] * 1)

    scorer = Scorer(y_true, y_pred)
    result = scorer.run_scores()

    expected_cm = np.array([[999999, 0, 0], [0, 0, 1], [0, 0, 0]])

    assert np.array_equal(result[0], expected_cm)
    assert result[1] == "/tmp/cm.png"
    assert result[2] == 1.0 / 3
    assert result[3] < 1.0
    assert result[3] > 0.5


def test_transform_to_numerical_large_set():
    """Test _transform_to_numerical with a large set of labels."""
    labels = [f"label_{i}" for i in range(1000)]
    y_true = np.array(labels)
    y_pred = np.array(labels)

    with pytest.raises(ValueError, match="y_true contains invalid labels"):
        Scorer(y_true, y_pred)
