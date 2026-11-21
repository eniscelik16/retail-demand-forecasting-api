import pandas as pd
import pytest

from retail_demand_forecasting.data import time_split


def make_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=10),
            "store": 1,
            "item": 1,
            "sales": range(10),
        }
    )


def test_train_is_strictly_before_test():
    train, test = time_split(make_df(), "2017-01-06")
    assert train["date"].max() < test["date"].min()


def test_no_rows_lost():
    df = make_df()
    train, test = time_split(df, "2017-01-06")
    assert len(train) + len(test) == len(df)


def test_cutoff_day_is_in_test():
    _, test = time_split(make_df(), "2017-01-06")
    assert test["date"].min() == pd.Timestamp("2017-01-06")


def test_cutoff_outside_data_raises():
    with pytest.raises(ValueError):
        time_split(make_df(), "2030-01-01")