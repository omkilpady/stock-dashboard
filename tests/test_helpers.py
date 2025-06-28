import importlib
import sys
import types
import os
import datetime as dt
import math

class FakeSeries:
    def __init__(self, values):
        self._values = list(values)

    @property
    def empty(self):
        return len(self._values) == 0

    class ILoc:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, idx):
            return self.values[idx]

    @property
    def iloc(self):
        return FakeSeries.ILoc(self._values)

class FakeData(dict):
    def __init__(self, values):
        super().__init__()
        self["Close"] = FakeSeries(values)

def load_helpers(fake_download):
    sys.modules['yfinance'] = types.SimpleNamespace(download=fake_download)
    if 'helpers' in sys.modules:
        del sys.modules['helpers']
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    return importlib.import_module('helpers')

def test_fx_to_usd_no_conversion():
    helpers = load_helpers(lambda *a, **k: FakeData([1]))
    assert helpers.fx_to_usd(10, 'USD') == 10

def test_fx_to_usd_with_rate():
    helpers = load_helpers(lambda *a, **k: FakeData([1.2]))
    assert helpers.fx_to_usd(10, 'EUR') == 12

def test_price_on_date_value():
    helpers = load_helpers(lambda *a, **k: FakeData([100, 101]))
    date = dt.date(2023, 1, 1)
    assert helpers.price_on_date('AAPL', date) == 100

def test_price_on_date_no_data():
    helpers = load_helpers(lambda *a, **k: FakeData([]))
    result = helpers.price_on_date('AAPL', dt.date(2023, 1, 1))
    assert math.isnan(result)
