"""Basit kural tabanlı baseline tahminciler.

Hepsi aynı imzayı kullanır:
    f(train, test) -> pd.Series (index = test.index)

`test` sadece hangi (date, store, item) için tahmin istediğimizi söyler;
test içindeki `sales` sütunu ASLA kullanılmaz (veri sızıntısı olmasın diye).
"""
import pandas as pd

KEYS = ["store", "item"]


def _attach(test: pd.DataFrame, table: pd.DataFrame, on: list[str], col: str) -> pd.Series:
    """test satırlarına, `on` sütunlarıyla eşleşen table[col] değerini ekler (sıra korunur)."""
    merged = test[on].merge(table[on + [col]], on=on, how="left")
    return pd.Series(merged[col].to_numpy(dtype=float), index=test.index, name="pred")


def naive_last(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Her mağaza-ürün için eğitimin son günündeki satışı tekrarlar."""
    last_day = train["date"].max()
    table = train.loc[train["date"] == last_day, KEYS + ["sales"]].rename(columns={"sales": "pred"})
    return _attach(test, table, KEYS, "pred")


def seasonal_naive_weekly(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Her test gününe, eğitimin son haftasındaki aynı haftanın gününün satışını verir."""
    last_week = train[train["date"] > train["date"].max() - pd.Timedelta(days=7)].copy()
    last_week["dow"] = last_week["date"].dt.day_of_week
    table = last_week[KEYS + ["dow", "sales"]].rename(columns={"sales": "pred"})
    test_keys = test[["date"] + KEYS].copy()
    test_keys["dow"] = test_keys["date"].dt.day_of_week
    return _attach(test_keys, table, KEYS + ["dow"], "pred")


def weekly_average(train: pd.DataFrame, test: pd.DataFrame, weeks: int = 4) -> pd.Series:
    """Son `weeks` haftanın, aynı haftanın günlerine ait ortalama satışı."""
    recent = train[train["date"] > train["date"].max() - pd.Timedelta(days=7 * weeks)].copy()
    recent["dow"] = recent["date"].dt.day_of_week
    table = recent.groupby(KEYS + ["dow"], as_index=False)["sales"].mean().rename(columns={"sales": "pred"})
    test_keys = test[["date"] + KEYS].copy()
    test_keys["dow"] = test_keys["date"].dt.day_of_week
    return _attach(test_keys, table, KEYS + ["dow"], "pred")


def yearly_seasonal_growth(train: pd.DataFrame, test: pd.DataFrame, growth_days: int = 91) -> pd.Series:
    """364 gün (52 hafta) önceki satış x son dönemin yıllık büyüme oranı.

    364 gün: haftanın günü aynı kalır. Büyüme oranı = eğitimin son `growth_days`
    gününün toplamı / bir yıl önceki aynı dönemin toplamı (mağaza-ürün bazında).
    """
    lag = pd.Timedelta(days=364)

    # Geçen yılın satışını test tarihine kaydır: (tarih + 364 gün) -> o günün satışı
    last_year = train[["date"] + KEYS + ["sales"]].copy()
    last_year["date"] = last_year["date"] + lag
    last_year = last_year.rename(columns={"sales": "ly_sales"})
    ly = _attach(test[["date"] + KEYS], last_year, ["date"] + KEYS, "ly_sales")
    if ly.isna().any():
        raise ValueError("Test tarihlerinin 364 gün öncesi eğitim verisinde yok (ufuk çok uzun).")

    # Büyüme oranı
    end = train["date"].max()
    start = end - pd.Timedelta(days=growth_days - 1)
    recent = train[(train["date"] >= start) & (train["date"] <= end)].groupby(KEYS)["sales"].sum()
    prior = train[(train["date"] >= start - lag) & (train["date"] <= end - lag)].groupby(KEYS)["sales"].sum()
    growth = (recent / prior).replace([float("inf")], float("nan")).fillna(1.0).rename("growth").reset_index()
    g = _attach(test[KEYS], growth, KEYS, "growth").fillna(1.0)

    return (ly * g).rename("pred")


BASELINES = {
    "naive_last": naive_last,
    "seasonal_naive_weekly": seasonal_naive_weekly,
    "weekly_average_4w": weekly_average,
    "yearly_seasonal_growth": yearly_seasonal_growth,
}