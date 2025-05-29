import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from environments.rewards.projected_returns import DelayedReturnsReward

class TestDelayedReturnsReward(unittest.TestCase):
    def setUp(self):
        # Create synthetic raw_data for 10 days, 2 assets, single index on 'day'
        self.days = np.arange(10)
        self.assets = ['A', 'B']
        data = []
        for day in self.days:
            for asset, price in zip(self.assets, [10 + day, 20 + 2*day]):
                data.append({'day': day, 'ticker': asset, 'close': price, 'open': price-1})
        self.raw_data = pd.DataFrame(data)
        self.raw_data.set_index('day', inplace=True)

    def get_prices(self, day, price_column):
        # Return prices for all assets in order [A, B] for a given day and price_column
        day_data = self.raw_data.loc[day]
        # If only one asset, day_data is a Series, else DataFrame
        if isinstance(day_data, pd.Series):
            return np.array([day_data[price_column]])
        return np.array([day_data[day_data['ticker'] == asset][price_column] for asset in self.assets]).flatten()

    def test_zero_reward_if_not_enough_data(self):
        reward = DelayedReturnsReward({'delay': 5, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * self.get_prices(8, 'close'))
        r = reward.calculate(self.raw_data, 8, pos, prev_val)

        new_val = np.sum(pos * self.get_prices(8+1, 'close'))
        expected = (new_val - prev_val) / prev_val
        self.assertEqual(r, expected)

    def test_correct_reward_after_delay(self):
        reward = DelayedReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 2])
        prev_val = np.sum(pos * self.get_prices(5, 'close'))
        r = reward.calculate(self.raw_data, 5, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(7, 'close'))
        expected = (new_val - prev_val) / prev_val
        self.assertAlmostEqual(r, expected)

    def test_delay_adjusts_at_end(self):
        reward = DelayedReturnsReward({'delay': 5, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * self.get_prices(7, 'close'))
        r = reward.calculate(self.raw_data, 7, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(9, 'close'))
        expected = (new_val - prev_val) / prev_val
        self.assertAlmostEqual(r, expected)

    def test_scaling(self):
        reward = DelayedReturnsReward({'delay': 2, 'scale': 2.0, 'price_column': 'close'})
        pos = np.array([1, 2])
        prev_val = np.sum(pos * self.get_prices(3, 'close'))
        r = reward.calculate(self.raw_data, 3, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(5, 'close'))
        expected = ((new_val - prev_val) / prev_val) * 2.0
        self.assertAlmostEqual(r, expected)

    def test_price_column(self):
        reward = DelayedReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'open'})
        pos = np.array([1, 2])
        prev_val = np.sum(pos * self.get_prices(2, 'open'))
        r = reward.calculate(self.raw_data, 2, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(4, 'open'))
        expected = (new_val - prev_val) / prev_val
        self.assertAlmostEqual(r, expected)

    def test_zero_position(self):
        reward = DelayedReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        pos = np.array([0, 0])
        prev_val = 0.0
        r = reward.calculate(self.raw_data, 2, pos, prev_val)
        self.assertEqual(r, 0.0)

    def test_negative_return(self):
        raw_data = self.raw_data.copy()
        # Set day 5 prices high, day 7 prices low
        raw_data.loc[5, 'close'] = 100
        raw_data.loc[7, 'close'] = 50

        reward = DelayedReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        
        pos = np.array([1, 1])
        prev_val = np.sum(pos * [100, 100])
        r = reward.calculate(raw_data, 5, pos, prev_val)
    
        new_val = np.sum(pos * [50, 50])
        expected = (new_val - prev_val) / prev_val
        self.assertAlmostEqual(r, expected)
        self.assertLess(r, 0)

    def test_reset(self):
        reward = DelayedReturnsReward({'delay': 2, 'scale': 1.0, 'price_column': 'close'})
        reward.reset()

if __name__ == '__main__':
    unittest.main() 