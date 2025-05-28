import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.agents.temperature_manager import TemperatureManager
from config.temperature import get_default_temperature_config

class TestTemperatureManager(unittest.TestCase):
    def setUp(self):
        self.total_env_steps = 1000
        self.warmup_steps = 100
        self.update_frequency = 10
        self.head_config = get_default_temperature_config()

    def test_initialization(self):
        manager = TemperatureManager(
            head_configs=self.head_config,
            update_frequency=self.update_frequency,
            total_env_steps=self.total_env_steps,
            warmup_steps=self.warmup_steps
        )
        self.assertEqual(manager.total_env_steps, self.total_env_steps)
        self.assertEqual(manager.warmup_steps, self.warmup_steps)
        self.assertEqual(manager.training_step, 0)
        self.assertEqual(manager.global_step, 0)
        self.assertEqual(manager.update_frequency, self.update_frequency)
        self.assertIn("discrete", set(manager.head_configs["head_configs"].keys()))
        self.assertIn("confidence", set(manager.head_configs["head_configs"].keys()))
        self.assertIn("value", set(manager.head_configs["head_configs"].keys()))

    def test_all_heads_temperatures(self):
        manager = TemperatureManager(
            head_configs=self.head_config,
            update_frequency=self.update_frequency,
            total_env_steps=self.total_env_steps,
            warmup_steps=self.warmup_steps
        )
        temps = manager.get_all_temperatures()
        self.assertIn("discrete", temps)
        self.assertIn("confidence", temps)
        self.assertIn("value", temps)
        self.assertTrue(all(isinstance(t, float) for t in temps.values()))

    def test_temperature_decay_and_warmup(self):
        manager = TemperatureManager(
            head_configs=self.head_config,
            update_frequency=self.update_frequency,
            total_env_steps=self.total_env_steps,
            warmup_steps=self.warmup_steps
        )
        temps_initial = manager.get_all_temperatures()
        # Step through warmup
        for _ in range(self.warmup_steps):
            manager.step()
        temps_warmup = manager.get_all_temperatures()
        # Step through decay
        for _ in range(200):
            manager.step()
        temps_decay = manager.get_all_temperatures()
        # Temperatures should decrease or stay the same after warmup
        for head in temps_initial:
            self.assertGreaterEqual(temps_warmup[head], temps_decay[head])

    def test_reset(self):
        manager = TemperatureManager(
            head_configs=self.head_config,
            update_frequency=self.update_frequency,
            total_env_steps=self.total_env_steps,
            warmup_steps=self.warmup_steps
        )
        for _ in range(50):
            manager.step()
        manager.reset()
        self.assertEqual(manager.training_step, 0)
        temps_reset = manager.get_all_temperatures()
        temps_initial = TemperatureManager(
            head_configs=self.head_config,
            update_frequency=self.update_frequency,
            total_env_steps=self.total_env_steps,
            warmup_steps=self.warmup_steps
        ).get_all_temperatures()
        for head in temps_initial:
            self.assertAlmostEqual(temps_reset[head], temps_initial[head], places=5)

    def test_progress_info(self):
        manager = TemperatureManager(
            head_configs=self.head_config,
            update_frequency=self.update_frequency,
            total_env_steps=self.total_env_steps,
            warmup_steps=self.warmup_steps
        )
        info = manager.get_progress_info()
        self.assertIn("training_step", info)
        self.assertIn("total_updates", info)
        self.assertIn("progress", info)
        self.assertIn("temperatures", info)
        self.assertIn("active_heads", info)
        self.assertIsInstance(info["temperatures"], dict)

if __name__ == "__main__":
    unittest.main()
