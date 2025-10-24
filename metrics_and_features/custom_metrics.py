from sklearn.metrics import roc_auc_score
import numpy as np


def recall_at_k(y_true, y_prob, k=0.1):
    """
    Calculates recall by labeling the top k% of predicted probabilities as positive.

    Parameters:
        y_true (list): True binary labels.
        y_prob (list): Predicted probabilities.
        k (float): Percentile of probabilities to label as positive (default 0.1).

    Returns:
        float: Recall rate in the top k% predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]

    tp_at_k = y_true[top].sum()
    P = y_true.sum()

    return float(tp_at_k / P) if P > 0 else 0.0


def lift_at_k(y_true, y_prob, k=0.1):
    """
    Calculates lift (precision/prevalence) by labeling the top k% of predicted probabilities as positive.

    Parameters:
        y_true (list): True binary labels.
        y_prob (list): Predicted probabilities.
        k (float): Percentile of probabilities to label as positive (default 0.1).

    Returns:
        float: Lift value in the top k% predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]

    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()

    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0


def convert_auc_to_gini(auc):
    """
    Converts ROC AUC score to Gini coefficient.

    The Gini coefficient is a linear transformation of the ROC AUC score.

    Parameters:
        auc (float): ROC AUC score (between 0 and 1).

    Returns:
        float: Gini coefficient (between -1 and 1).
    """
    return 2 * auc - 1


def ing_hubs_datathon_metric(y_true, y_prob):
    """
    Calculates a custom metric that combines Gini, recall@10%, and lift@10% metrics.

    The metric ratios each score against a baseline model's metric values and applies the following weights:
    - Gini: 40%
    - Recall@10%: 30%
    - Lift@10%: 30%

    Parameters:
        y_true (list): True binary labels.
        y_prob (list): Predicted probabilities.

    Returns:
        float: Weighted composite score.
    """
    # Weights for the final metric
    score_weights = {
        "gini": 0.4,
        "recall_at_10perc": 0.3,
        "lift_at_10perc": 0.3,
    }

    # Baseline model's metric values
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }

    # Calculate metrics for y_prob predictions
    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)

    new_scores = {
        "roc_auc": roc_auc,
        "recall_at_10perc": recall_at_10perc,
        "lift_at_10perc": lift_at_10perc,
    }

    # Convert ROC AUC values to Gini coefficient
    baseline_scores["gini"] = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_scores["gini"] = convert_auc_to_gini(new_scores["roc_auc"])

    # Ratio against baseline model
    final_gini_score = new_scores["gini"] / baseline_scores["gini"]
    final_recall_score = new_scores["recall_at_10perc"] / baseline_scores["recall_at_10perc"]
    final_lift_score = new_scores["lift_at_10perc"] / baseline_scores["lift_at_10perc"]

    # Calculate weighted metric
    final_score = (
        final_gini_score * score_weights["gini"] +
        final_recall_score * score_weights["recall_at_10perc"] + 
        final_lift_score * score_weights["lift_at_10perc"]
    )
    return final_score