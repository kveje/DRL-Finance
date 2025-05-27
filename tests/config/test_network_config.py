import unittest
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.networks import get_network_config, BASE_CONFIG

class TestNetworkConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.base_config = BASE_CONFIG.copy()

    def test_default_config(self):
        """Test default configuration generation"""
        config = get_network_config(n_assets=10, window_size=10)
        
        # Check that all processors are present
        self.assertIn('price', config['processors'])
        self.assertIn('ohlcv', config['processors'])
        self.assertIn('technical', config['processors'])
        self.assertIn('position', config['processors'])
        self.assertIn('cash', config['processors'])
        self.assertIn('affordability', config['processors'])
        self.assertIn('current_price', config['processors'])
        
        # Check default head configuration
        self.assertTrue(config['heads']['discrete']['enabled'])
        self.assertFalse(config['heads']['confidence']['enabled'])
        self.assertFalse(config['heads']['value']['enabled'])
        self.assertEqual(config['heads']['discrete']['type'], 'parametric')

    def test_price_type_configurations(self):
        """Test different price type configurations"""
        # Test price only
        price_config = get_network_config(price_type='price', n_assets=10, window_size=10)
        self.assertFalse(price_config['processors']['ohlcv']['enabled'])
        self.assertTrue(price_config['processors']['price']['enabled'])
        
        # Test OHLCV only
        ohlcv_config = get_network_config(price_type='ohlcv', n_assets=10, window_size=10)
        self.assertFalse(ohlcv_config['processors']['price']['enabled'])
        self.assertTrue(ohlcv_config['processors']['ohlcv']['enabled'])
        
        # Test both
        both_config = get_network_config(price_type='both', n_assets=10, window_size=10)
        self.assertTrue(both_config['processors']['price']['enabled'])
        self.assertTrue(both_config['processors']['ohlcv']['enabled'])

    def test_technical_dim_configuration(self):
        """Test technical dimension configuration"""
        # Test with technical indicators
        tech_config = get_network_config(technical_dim=10, n_assets=10, window_size=10)
        self.assertTrue(tech_config['processors']['technical']['enabled'])
        self.assertEqual(tech_config['processors']['technical']['tech_dim'], 10)
        
        # Test without technical indicators
        no_tech_config = get_network_config(technical_dim=None, n_assets=10, window_size=10)
        self.assertFalse(no_tech_config['processors']['technical']['enabled'])

    def test_head_type_configurations(self):
        """Test different head type configurations"""
        # Test parametric head
        parametric_config = get_network_config(head_type='parametric', n_assets=10, window_size=10)
        self.assertEqual(parametric_config['heads']['discrete']['type'], 'parametric')
        
        # Test bayesian head
        bayesian_config = get_network_config(head_type='bayesian', n_assets=10, window_size=10)
        self.assertEqual(bayesian_config['heads']['discrete']['type'], 'bayesian')

    def test_discrete_head_configuration(self):
        """Test discrete head configuration"""
        # Test with discrete head enabled (default)
        enabled_config = get_network_config(include_discrete=True, n_assets=10, window_size=10)
        self.assertTrue(enabled_config['heads']['discrete']['enabled'])
        
        # Test with discrete head disabled
        disabled_config = get_network_config(include_discrete=False, n_assets=10, window_size=10)
        self.assertFalse(disabled_config['heads']['discrete']['enabled'])

    def test_confidence_and_value_heads(self):
        """Test confidence and value head configurations"""
        # Test with both heads
        full_config = get_network_config(
            include_confidence=True,
            include_value=True,
            n_assets=10,
            window_size=10
        )
        self.assertTrue(full_config['heads']['confidence']['enabled'])
        self.assertTrue(full_config['heads']['value']['enabled'])
        
        # Test with only confidence
        confidence_config = get_network_config(include_confidence=True, n_assets=10, window_size=10)
        self.assertTrue(confidence_config['heads']['confidence']['enabled'])
        self.assertFalse(confidence_config['heads']['value']['enabled'])
        
        # Test with only value
        value_config = get_network_config(include_value=True, n_assets=10, window_size=10)
        self.assertFalse(value_config['heads']['confidence']['enabled'])
        self.assertTrue(value_config['heads']['value']['enabled'])

    def test_head_dimensions(self):
        """Test head dimension calculations"""
        # Test large network
        large_config = get_network_config(price_type='both', technical_dim=50, n_assets=10, window_size=10)
        self.assertEqual(large_config['heads']['discrete']['hidden_dim'], 64)
        self.assertEqual(large_config['heads']['confidence']['hidden_dim'], 32)
        self.assertEqual(large_config['heads']['value']['hidden_dim'], 16)
        
        # Test medium network
        medium_config = get_network_config(price_type='price', technical_dim=20, n_assets=10, window_size=10)
        self.assertEqual(medium_config['heads']['discrete']['hidden_dim'], 64)
        self.assertEqual(medium_config['heads']['confidence']['hidden_dim'], 32)
        self.assertEqual(medium_config['heads']['value']['hidden_dim'], 16)

    def test_bayesian_head_configuration(self):
        """Test bayesian head configuration with confidence and value"""
        config = get_network_config(
            head_type='bayesian',
            include_confidence=True,
            include_value=True,
            n_assets=10,
            window_size=10
        )
        self.assertEqual(config['heads']['discrete']['type'], 'bayesian')
        self.assertEqual(config['heads']['confidence']['type'], 'bayesian')
        self.assertEqual(config['heads']['value']['type'], 'bayesian')

if __name__ == '__main__':
    unittest.main() 