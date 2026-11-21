from pathlib import Path

import pandas as pd


def load_sales(path: str | Path) -> pd.DataFrame:
    """CSV'yi okur, tarihi tarih tipine çevirir, mağaza-ürün-tarih sırasına dizer."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Veriyi kesme tarihinden ikiye böler.

    train: cutoff'tan ÖNCEKİ günler
    test : cutoff günü ve sonrası
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df["date"] < cutoff_ts].copy()
    test = df[df["date"] >= cutoff_ts].copy()
    if train.empty or test.empty:
        raise ValueError(f"cutoff={cutoff} verinin dışında, train veya test boş kaldı.")
    return train, test