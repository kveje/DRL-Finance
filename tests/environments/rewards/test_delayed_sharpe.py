import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) )
from environments.rewards.projected_sharpe import DelayedSharpeReward

class TestDelayedSharpeReward(unittest.TestCase):
    def setUp(self):
        self.days = np.arange(10)
        self.assets = ['A', 'B']
        data = []
        for day in self.days:
            for asset, price in zip(self.assets, [10 + day, 20 + 2*day]):
                data.append({'day': day, 'ticker': asset, 'close': price, 'open': price-1})
        self.raw_data = pd.DataFrame(data)
        self.raw_data.set_index('day', inplace=True)
        self.annualization_factor = 252
        self.annual_risk_free_rate = 0.02
        self.daily_risk_free_rate = (1 + self.annual_risk_free_rate) ** (1/self.annualization_factor) - 1
        self.min_history_size = 2

    def get_prices(self, day, price_column):
        day_data = self.raw_data.loc[day]
        if isinstance(day_data, pd.Series):
            return np.array([day_data[price_column]])
        return np.array([day_data[day_data['ticker'] == asset][price_column] for asset in self.assets]).flatten()

    def test_zero_reward_if_not_enough_history(self):
        reward = DelayedSharpeReward({'delay': 2, 'window_size': 3, 'scale': 1.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * self.get_prices(0, 'close'))
        r = reward.calculate(self.raw_data, 0, pos, prev_val)
        # Only one return, so should use the single-period logic
        delayed_return = (np.sum(pos * self.get_prices(2, 'close')) - prev_val) / prev_val
        expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
        self.assertAlmostEqual(r, expected)

    def test_sharpe_after_enough_steps(self):
        reward = DelayedSharpeReward({'delay': 1, 'window_size': 3, 'scale': 1.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([1, 1])
        returns = []
        for day in range(6):
            prev_val = np.sum(pos * self.get_prices(day, 'close'))
            r = reward.calculate(self.raw_data, day, pos, prev_val)
            if day == 0:
                # First call, only one return, use single-period logic
                delayed_return = (np.sum(pos * self.get_prices(day+1, 'close')) - prev_val) / prev_val
                expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
                self.assertAlmostEqual(r, expected)
            else:
                new_val = np.sum(pos * self.get_prices(day+1, 'close'))
                returns.append((new_val - prev_val) / prev_val)
                if len(returns) >= self.min_history_size:
                    arr = np.array(returns[-3:])
                    excess_returns = arr - self.daily_risk_free_rate
                    expected = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(self.annualization_factor)
                    self.assertAlmostEqual(r, expected)
                elif len(returns) > 0:
                    delayed_return = returns[-1]
                    expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
                    self.assertAlmostEqual(r, expected)

    def test_delay_adjusts_at_end(self):
        reward = DelayedSharpeReward({'delay': 5, 'window_size': 2, 'scale': 1.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([1, 1])
        # Fill history
        for day in range(7):
            prev_val = np.sum(pos * self.get_prices(day, 'close'))
            reward.calculate(self.raw_data, day, pos, prev_val)
        # Now test at day 7 (delay will be 2)
        prev_val = np.sum(pos * self.get_prices(7, 'close'))
        r = reward.calculate(self.raw_data, 7, pos, prev_val)
        new_val = np.sum(pos * self.get_prices(9, 'close'))
        delayed_return = (new_val - prev_val) / prev_val
        arr = np.array([reward.returns_history[-2], reward.returns_history[-1]])
        excess_returns = arr - self.daily_risk_free_rate
        expected = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(self.annualization_factor)
        self.assertAlmostEqual(r, expected)

    def test_scaling(self):
        reward = DelayedSharpeReward({'delay': 1, 'window_size': 3, 'scale': 2.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([1, 1])
        returns = []
        for day in range(4):
            prev_val = np.sum(pos * self.get_prices(day, 'close'))
            r = reward.calculate(self.raw_data, day, pos, prev_val)
            if day == 0:
                delayed_return = (np.sum(pos * self.get_prices(day+1, 'close')) - prev_val) / prev_val
                expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor) * 2.0
                self.assertAlmostEqual(r, expected)
            else:
                new_val = np.sum(pos * self.get_prices(day+1, 'close'))
                returns.append((new_val - prev_val) / prev_val)
                if len(returns) >= self.min_history_size:
                    arr = np.array(returns[-3:])
                    excess_returns = arr - self.daily_risk_free_rate
                    expected = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)) * np.sqrt(self.annualization_factor) * 2.0
                    self.assertAlmostEqual(r, expected)
                elif len(returns) > 0:
                    delayed_return = returns[-1]
                    expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor) * 2.0
                    self.assertAlmostEqual(r, expected)

    def test_price_column(self):
        reward = DelayedSharpeReward({'delay': 1, 'window_size': 3, 'scale': 1.0, 'price_column': 'open',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([1, 1])
        returns = []
        for day in range(4):
            prev_val = np.sum(pos * self.get_prices(day, 'open'))
            r = reward.calculate(self.raw_data, day, pos, prev_val)
            if day == 0:
                delayed_return = (np.sum(pos * self.get_prices(day+1, 'open')) - prev_val) / prev_val
                expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
                self.assertAlmostEqual(r, expected)
            else:
                new_val = np.sum(pos * self.get_prices(day+1, 'open'))
                returns.append((new_val - prev_val) / prev_val)
                if len(returns) >= self.min_history_size:
                    arr = np.array(returns[-3:])
                    excess_returns = arr - self.daily_risk_free_rate
                    expected = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(self.annualization_factor)
                    self.assertAlmostEqual(r, expected)
                elif len(returns) > 0:
                    delayed_return = returns[-1]
                    expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
                    self.assertAlmostEqual(r, expected)

    def test_zero_position(self):
        reward = DelayedSharpeReward({'delay': 1, 'window_size': 3, 'scale': 1.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([0, 0])
        prev_val = 0.0
        r = reward.calculate(self.raw_data, 2, pos, prev_val)
        self.assertEqual(r, 0.0)

    def test_negative_or_zero_return(self):
        raw_data = self.raw_data.copy()
        raw_data.loc[5, 'close'] = 100
        raw_data.loc[7, 'close'] = 0
        reward = DelayedSharpeReward({'delay': 2, 'window_size': 3, 'scale': 1.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        pos = np.array([1, 1])
        prev_val = np.sum(pos * [100, 100])
        r = reward.calculate(raw_data, 5, pos, prev_val)
        # If the return is zero (because new_val is zero), expected is (delayed_return - daily_risk_free_rate) * sqrt(annualization_factor)
        delayed_return = (0 - prev_val) / prev_val if prev_val != 0 else 0
        expected = (delayed_return - self.daily_risk_free_rate) * np.sqrt(self.annualization_factor)
        self.assertAlmostEqual(r, expected)

    def test_reset(self):
        reward = DelayedSharpeReward({'delay': 1, 'window_size': 3, 'scale': 1.0, 'price_column': 'close',
                                      'annualization_factor': self.annualization_factor,
                                      'annual_risk_free_rate': self.annual_risk_free_rate,
                                      'min_history_size': self.min_history_size})
        reward.reset()

if __name__ == '__main__':
    unittest.main() 