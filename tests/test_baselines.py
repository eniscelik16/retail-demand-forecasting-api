import pandas as pd
import pytest

from retail_demand_forecasting.baselines import (
    BASELINES,
    naive_last,
    seasonal_naive_weekly,
    weekly_average,
    yearly_seasonal_growth,
)


def make_series(start, days, value_fn, store=1, item=1):
    dates = pd.date_range(start, periods=days)
    return pd.DataFrame(
        {"date": dates, "store": store, "item": item, "sales": [value_fn(i, d) for i, d in enumerate(dates)]}
    )


def split(df, n_test):
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def test_naive_last_repeats_last_value():
    df = make_series("2017-01-01", 10, lambda i, d: i)
    train, test = split(df, 3)
    pred = naive_last(train, test)
    assert (pred == 6).all()  # eğitimin son günü i=6


def test_seasonal_naive_weekly_repeats_weekday_pattern():
    df = make_series("2017-01-02", 35, lambda i, d: d.day_of_week * 10)
    train, test = split(df, 7)
    pred = seasonal_naive_weekly(train, test)
    assert pred.tolist() == test["sales"].tolist()


def test_weekly_average_averages_same_weekdays():
    # Pazartesiler: 10, 20, 30, 40 -> ortalama 25; diğer günler 0
    df = make_series("2017-01-02", 35, lambda i, d: (i // 7 + 1) * 10 if d.day_of_week == 0 else 0)
    train, test = split(df, 7)
    pred = weekly_average(train, test, weeks=4)
    assert pred.iloc[0] == pytest.approx(25)  # test'in ilk günü pazartesi
    assert (pred.iloc[1:] == 0).all()


def test_yearly_seasonal_growth_scales_last_year_by_growth():
    # İlk 364 gün satış 10, sonrası 20 -> büyüme x2, geçen yıl 10 -> tahmin 20
    df = make_series("2016-01-04", 607, lambda i, d: 10 if i < 364 else 20)
    train, test = split(df, 7)
    pred = yearly_seasonal_growth(train, test)
    assert pred.tolist() == pytest.approx([20] * 7)


def test_yearly_seasonal_growth_raises_if_horizon_too_long():
    df = make_series("2017-01-01", 100, lambda i, d: 5)
    train, test = split(df, 7)
    with pytest.raises(ValueError):
        yearly_seasonal_growth(train, test)


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_do_not_use_test_sales(name):
    df = make_series("2016-01-04", 607, lambda i, d: (i % 7) + i // 100)
    train, test = split(df, 7)
    f = BASELINES[name]
    pred_a = f(train, test)
    pred_b = f(train, test.assign(sales=999_999))
    pd.testing.assert_series_equal(pred_a, pred_b)


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_return_one_prediction_per_test_row(name):
    # iki seri, karışık sıra
    a = make_series("2016-01-04", 607, lambda i, d: i % 7, store=1, item=1)
    b = make_series("2016-01-04", 607, lambda i, d: 50 + i % 7, store=2, item=3)
    df = pd.concat([a, b]).sort_values(["date", "store"]).reset_index(drop=True)
    train = df[df["date"] < "2017-08-28"]
    test = df[df["date"] >= "2017-08-28"]
    pred = BASELINES[name](train, test)
    assert len(pred) == len(test)
    assert pred.index.equals(test.index)
    assert not pred.isna().any()