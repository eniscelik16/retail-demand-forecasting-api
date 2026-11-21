import pytest

from retail_demand_forecasting.metrics import bias, mae, wape

# Elle hesap: gerçek=[10,20,30], tahmin=[12,18,36]
# hatalar: 2, 2, 6  -> toplam 10
# MAE  = 10/3     = 3.3333
# WAPE = 10/60    = 0.16667
# bias = 66/60 -1 = 0.1
Y_TRUE = [10, 20, 30]
Y_PRED = [12, 18, 36]


def test_mae():
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(10 / 3)


def test_wape():
    assert wape(Y_TRUE, Y_PRED) == pytest.approx(10 / 60)


def test_bias():
    assert bias(Y_TRUE, Y_PRED) == pytest.approx(0.1)


def test_perfect_forecast_has_zero_error():
    assert mae(Y_TRUE, Y_TRUE) == 0
    assert wape(Y_TRUE, Y_TRUE) == 0
    assert bias(Y_TRUE, Y_TRUE) == 0


def test_different_lengths_raise():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


def test_zero_total_actual_raises():
    with pytest.raises(ValueError):
        wape([0, 0], [1, 2])