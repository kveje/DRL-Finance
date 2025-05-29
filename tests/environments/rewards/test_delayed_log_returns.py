import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from environments.rewards.projected_log_returns import DelayedLogReturnsReward

class TestDelayedLogReturnsReward(unittest.TestCase):
    def setUp(self):
        self.days = np.arange(10)
        self.assets = ['A', 'B']
        data = []
        for day in self.days:
            for asset, price in zip(self.assets, [10 + day, 20 + 2*day]):
                data.append({'day': day, 'ticker': asset, 'close': price, 'open': price-1})
        self.raw_data = pd.DataFrame(data)
        self.raw_data.set_index('day', inplace=True)

    def get_prices(self, day, price_column):
        day_data = self.raw_data.loc[day]
        if isinstance(day_data, pd.Series):
            return np.array([day_data[price_column]])
        return np.array([day_data[day_data['ticker'] == asset][price_column] for asset in self.assets]).flatten()

    def test_zero_reward_if_not_enough_data(self):
        reward = DelayedLogReturnsReward({'delay': 5, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * self.get_prices(8, 'close'))
        r = reward.calculate(self.raw_data, 8, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(9, 'close'))
        expected = np.log(new_val / prev_val) if prev_val > 0 and new_val > 0 else 0.0
        self.assertEqual(r, expected)

    def test_correct_log_return_after_delay(self):
        reward = DelayedLogReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 2])
        prev_val = np.sum(pos * self.get_prices(5, 'close'))
        r = reward.calculate(self.raw_data, 5, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(7, 'close'))
        expected = np.log(new_val / prev_val)
        self.assertAlmostEqual(r, expected)

    def test_delay_adjusts_at_end(self):
        reward = DelayedLogReturnsReward({'delay': 5, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * self.get_prices(7, 'close'))
        r = reward.calculate(self.raw_data, 7, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(9, 'close'))
        expected = np.log(new_val / prev_val)
        self.assertAlmostEqual(r, expected)

    def test_scaling(self):
        reward = DelayedLogReturnsReward({'delay': 2, 'scale': 2.0, 'price_column': 'close'})
        pos = np.array([1, 2])
        prev_val = np.sum(pos * self.get_prices(3, 'close'))
        r = reward.calculate(self.raw_data, 3, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(5, 'close'))
        expected = np.log(new_val / prev_val) * 2.0
        self.assertAlmostEqual(r, expected)

    def test_price_column(self):
        reward = DelayedLogReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'open'})
        pos = np.array([1, 2])
        prev_val = np.sum(pos * self.get_prices(2, 'open'))
        r = reward.calculate(self.raw_data, 2, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(4, 'open'))
        expected = np.log(new_val / prev_val)
        self.assertAlmostEqual(r, expected)

    def test_zero_position(self):
        reward = DelayedLogReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([0, 0])
        prev_val = 0.0
        r = reward.calculate(self.raw_data, 2, pos, prev_val)
        self.assertEqual(r, 0.0)

    def test_negative_or_zero_return(self):
        raw_data = self.raw_data.copy()
        raw_data.loc[5, 'close'] = 100
        raw_data.loc[7, 'close'] = 0  # zero price
        reward = DelayedLogReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * [100, 100])
        r = reward.calculate(raw_data, 5, pos, prev_val)
        new_val = np.sum(pos * [0, 0])
        expected = 0.0  # log(0/prev_val) is not defined, should return 0
        self.assertEqual(r, expected)

    def test_reset(self):
        reward = DelayedLogReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        reward.reset()

if __name__ == '__main__':
    unittest.main() 