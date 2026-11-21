import numpy as np


def _to_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true ve y_pred aynı uzunlukta olmalı.")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Ortalama mutlak hata: ortalama kaç adet yanılıyoruz."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred) -> float:
    """Toplam mutlak hata / toplam gerçek satış (0.15 = %15)."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    total = y_true.sum()
    if total == 0:
        raise ValueError("Toplam gerçek satış 0, WAPE hesaplanamaz.")
    return float(np.abs(y_true - y_pred).sum() / total)


def bias(y_true, y_pred) -> float:
    """Toplam tahmin / toplam gerçek - 1. Pozitif: fazla tahmin, negatif: az tahmin."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    total = y_true.sum()
    if total == 0:
        raise ValueError("Toplam gerçek satış 0, bias hesaplanamaz.")
    return float(y_pred.sum() / total - 1)