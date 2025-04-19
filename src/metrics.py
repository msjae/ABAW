import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

def ccc_score(y_pred: list, y_true: list) -> float:
    """Calculate Concordance Correlation Coefficient (CCC)."""
    epsilon = 1e-10
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    cor = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)) + epsilon)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2

    return numerator / denominator

def one_hot_transfer(label: list, class_num: int) -> np.ndarray:
    """Convert label to one-hot encoded array."""
    return np.eye(class_num)[label]

class ExprMetric:
    def __call__(self, pred: np.ndarray, gt: np.ndarray, class_num: int = 8) -> dict:
        """
        Calculate EXPR metrics (F1_macro, F1_weighted, ACC).
        """

        pred = np.argmax(pred, axis=1)

        acc = accuracy_score(gt, pred)
        f1_macro = f1_score(gt, pred, average='macro')
        f1_weighted = f1_score(gt, pred, average='weighted')

        return {'F1_macro': f1_macro, 'F1_weighted': f1_weighted, 'ACC': acc}

class VAMetric:
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """
        Calculate VA metrics using CCC for valence and arousal.
        """
        pred_v = pred[:, 0].flatten().tolist()
        pred_a = pred[:, 1].flatten().tolist()
        gt_v = gt[:, 0].flatten().tolist()
        gt_a = gt[:, 1].flatten().tolist()

        ccc_v = ccc_score(pred_v, gt_v)
        ccc_a = ccc_score(pred_a, gt_a)

        return {'valence_ccc': ccc_v, 'arousal_ccc': ccc_a, 'va_metric': 0.5 * (ccc_v + ccc_a)}

class AUMetric:
    def __call__(self, pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Calculate AU metrics (F1_macro, F1_weighted, ACC) using sigmoid thresholding.
        """
        pred = (pred > threshold).astype(int)

        acc = accuracy_score(gt, pred)
        f1_macro = f1_score(gt, pred, average='macro', zero_division=0)
        f1_weighted = f1_score(gt, pred, average='weighted', zero_division=0)

        return {'F1_macro': f1_macro, 'F1_weighted': f1_weighted, 'ACC': acc}