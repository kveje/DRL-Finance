"""
Diagnostic Plotter

This module provides tools for creating diagnostic plots that are useful
for theoretical analysis and research, including decision boundaries,
activation patterns, and comparative analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from models.agents.base_agent import BaseAgent
from environments.trading_env import TradingEnv


class DiagnosticPlotter:
    """
    Creates diagnostic plots for theoretical analysis.
    
    Provides tools for understanding network behavior through various
    visualization techniques useful for research and thesis work.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        train_env: TradingEnv,
        val_env: TradingEnv,
        experiment_data: Dict[str, Any],
        output_dir: Path,
        logger
    ):
        """
        Initialize the diagnostic plotter.
        
        Args:
            agent: Trained agent with neural network
            train_env: Training environment
            val_env: Validation environment
            experiment_data: Loaded experiment data
            output_dir: Directory to save visualizations
            logger: Logger instance
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.experiment_data = experiment_data
        self.output_dir = output_dir
        self.logger = logger
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get network and device
        self.network = agent.get_model()
        self.device = next(self.network.parameters()).device
        
        # Set network to evaluation mode
        self.network.eval()
        
        # Extract metrics if available
        self.metrics = experiment_data.get("metrics", {})
    
    def create_all_diagnostics(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Create all diagnostic plots.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing diagnostic analysis results
        """
        results = {}
        
        # Create decision boundary analysis
        results["decision_boundaries"] = self.analyze_decision_boundaries()
        
        # Create activation pattern analysis
        results["activation_patterns"] = self.analyze_activation_patterns()
        
        # Create learning progression analysis
        results["learning_progression"] = self.analyze_learning_progression()
        
        # Create comparative scenario analysis
        results["scenario_analysis"] = self.analyze_different_scenarios()
        
        # Create network behavior clustering
        results["behavior_clustering"] = self.analyze_behavior_clustering()
        
        if save_plots:
            self.plot_decision_boundaries(results["decision_boundaries"])
            self.plot_activation_patterns(results["activation_patterns"])
            self.plot_learning_progression(results["learning_progression"])
            self.plot_scenario_comparison(results["scenario_analysis"])
            self.plot_behavior_clustering(results["behavior_clustering"])
            self.create_interactive_dashboard(results)
        
        return results
    
    def analyze_decision_boundaries(self) -> Dict[str, Any]:
        """Analyze decision boundaries in different market conditions."""
        boundary_analysis = {}
        
        # Create a grid of market conditions
        market_conditions = self._create_market_condition_grid()
        
        # Analyze decisions across different conditions
        decisions = []
        conditions = []
        
        for condition in market_conditions:
            # Create observation based on condition
            obs = self._create_observation_from_condition(condition)
            
            # Get network decision
            with torch.no_grad():
                output = self.network(obs)
                action, _ = self.agent.get_intended_action(obs)
            
            decisions.append(action)
            conditions.append(condition)
        
        boundary_analysis["conditions"] = conditions
        boundary_analysis["decisions"] = decisions
        boundary_analysis["market_grid"] = market_conditions
        
        # Analyze decision patterns
        decision_array = np.array(decisions)
        boundary_analysis["decision_statistics"] = {
            "mean_action": np.mean(decision_array, axis=0),
            "std_action": np.std(decision_array, axis=0),
            "action_range": np.ptp(decision_array, axis=0),
            "decision_clusters": self._cluster_decisions(decision_array)
        }
        
        return boundary_analysis
    
    def analyze_activation_patterns(self) -> Dict[str, Any]:
        """Analyze activation patterns across different layers."""
        activation_analysis = {}
        
        # Hook to capture activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu().numpy()
                elif isinstance(output, dict):
                    for key, tensor in output.items():
                        if isinstance(tensor, torch.Tensor):
                            activations[f"{name}_{key}"] = tensor.detach().cpu().numpy()
            return hook
        
        # Register hooks on key layers
        for name, module in self.network.named_modules():
            if any(layer_type in name for layer_type in ['linear', 'conv', 'lstm']):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Get sample observations
        sample_obs = []
        for _ in range(50):
            obs = self.train_env.reset()
            tensor_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
            sample_obs.append(tensor_obs)
        
        # Collect activations
        all_activations = []
        for obs in sample_obs:
            activations.clear()
            with torch.no_grad():
                _ = self.network(obs)
            all_activations.append(activations.copy())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activation patterns
        activation_analysis["layer_statistics"] = {}
        for layer_name in all_activations[0].keys():
            layer_activations = [act[layer_name] for act in all_activations if layer_name in act]
            if layer_activations:
                stacked_activations = np.concatenate(layer_activations, axis=0)
                activation_analysis["layer_statistics"][layer_name] = {
                    "mean_activation": np.mean(stacked_activations),
                    "std_activation": np.std(stacked_activations),
                    "sparsity": np.mean(stacked_activations == 0),
                    "activation_distribution": stacked_activations.flatten()[:1000]  # Sample for plotting
                }
        
        return activation_analysis
    
    def analyze_learning_progression(self) -> Dict[str, Any]:
        """Analyze how the network's behavior changed during training."""
        progression_analysis = {}
        
        # Use metrics from experiment data if available
        if self.metrics:
            progression_analysis["training_metrics"] = self.metrics
            
            # Analyze metric trends
            if "sharpe_ratio" in self.metrics:
                sharpe_values = self.metrics["sharpe_ratio"]
                progression_analysis["sharpe_trend"] = {
                    "values": sharpe_values,
                    "trend_slope": np.polyfit(range(len(sharpe_values)), sharpe_values, 1)[0] if len(sharpe_values) > 1 else 0,
                    "volatility": np.std(sharpe_values) if len(sharpe_values) > 1 else 0
                }
            
            if "total_return" in self.metrics:
                return_values = self.metrics["total_return"]
                progression_analysis["return_trend"] = {
                    "values": return_values,
                    "cumulative_improvement": return_values[-1] - return_values[0] if len(return_values) > 1 else 0,
                    "learning_rate": np.mean(np.diff(return_values)) if len(return_values) > 1 else 0
                }
        
        # Analyze current network state vs random initialization
        progression_analysis["network_evolution"] = self._analyze_network_evolution()
        
        return progression_analysis
    
    def analyze_different_scenarios(self) -> Dict[str, Any]:
        """Analyze network behavior in different market scenarios."""
        scenario_analysis = {}
        
        # Define different market scenarios
        scenarios = {
            "bull_market": {"trend": 0.1, "volatility": 0.02, "correlation": 0.3},
            "bear_market": {"trend": -0.1, "volatility": 0.03, "correlation": 0.5},
            "high_volatility": {"trend": 0.0, "volatility": 0.05, "correlation": 0.1},
            "low_volatility": {"trend": 0.0, "volatility": 0.01, "correlation": 0.8},
            "crisis": {"trend": -0.2, "volatility": 0.08, "correlation": 0.9}
        }
        
        for scenario_name, params in scenarios.items():
            # Generate synthetic data for this scenario
            scenario_data = self._generate_scenario_data(params)
            
            # Analyze network behavior
            scenario_results = self._analyze_scenario_behavior(scenario_data)
            scenario_analysis[scenario_name] = scenario_results
        
        # Compare scenarios
        scenario_analysis["comparison"] = self._compare_scenarios(scenario_analysis)
        
        return scenario_analysis
    
    def analyze_behavior_clustering(self) -> Dict[str, Any]:
        """Cluster network behaviors to identify distinct patterns."""
        clustering_analysis = {}
        
        # Collect network outputs for different inputs
        outputs = []
        inputs = []
        
        for _ in range(200):
            obs = self.train_env.reset()
            tensor_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.network(tensor_obs)
                action, _ = self.agent.get_intended_action(tensor_obs)
            
            # Flatten output for clustering
            output_flat = []
            for tensor in output.values():
                output_flat.extend(tensor.flatten().cpu().numpy())
            
            outputs.append(output_flat)
            inputs.append(self._flatten_observation(tensor_obs))
        
        outputs = np.array(outputs)
        inputs = np.array(inputs)
        
        # Check for sufficient data diversity before clustering
        unique_outputs = np.unique(outputs, axis=0)
        n_unique = len(unique_outputs)
        
        # Adaptive number of clusters with better handling
        n_clusters = min(5, max(2, n_unique // 5))  # Ensure at least 2 clusters if we have unique data
        
        if n_unique < 2:
            # Handle case with insufficient diversity
            self.logger.warning("Insufficient data diversity for clustering analysis")
            clustering_analysis["clusters"] = np.zeros(len(outputs))
            clustering_analysis["cluster_centers"] = outputs[:1] if len(outputs) > 0 else np.array([])
            clustering_analysis["n_clusters"] = 1
            clustering_analysis["warning"] = "Insufficient data diversity for meaningful clustering"
        else:
            try:
                # Perform clustering with improved parameters
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42,
                    n_init=10,  # Multiple initializations
                    max_iter=300,  # More iterations
                    tol=1e-4  # Convergence tolerance
                )
                clusters = kmeans.fit_predict(outputs)
                
                clustering_analysis["clusters"] = clusters
                clustering_analysis["cluster_centers"] = kmeans.cluster_centers_
                clustering_analysis["n_clusters"] = n_clusters
                clustering_analysis["inertia"] = kmeans.inertia_
                
            except Exception as e:
                self.logger.warning(f"Clustering failed: {e}. Using single cluster.")
                clustering_analysis["clusters"] = np.zeros(len(outputs))
                clustering_analysis["cluster_centers"] = outputs[:1] if len(outputs) > 0 else np.array([])
                clustering_analysis["n_clusters"] = 1
                clustering_analysis["error"] = str(e)
        
        # Analyze cluster characteristics
        clustering_analysis["cluster_analysis"] = {}
        clusters = clustering_analysis["clusters"]
        n_clusters = clustering_analysis["n_clusters"]
        
        for i in range(n_clusters):
            cluster_mask = clusters == i
            if np.any(cluster_mask):  # Only analyze if cluster has members
                cluster_outputs = outputs[cluster_mask]
                cluster_inputs = inputs[cluster_mask]
                
                clustering_analysis["cluster_analysis"][i] = {
                    "size": np.sum(cluster_mask),
                    "output_mean": np.mean(cluster_outputs, axis=0) if len(cluster_outputs) > 0 else np.array([]),
                    "output_std": np.std(cluster_outputs, axis=0) if len(cluster_outputs) > 0 else np.array([]),
                    "input_characteristics": np.mean(cluster_inputs, axis=0) if len(cluster_inputs) > 0 else np.array([])
                }
        
        # Dimensionality reduction for visualization
        if outputs.shape[1] > 2 and len(outputs) > 2:
            try:
                pca = PCA(n_components=2)
                outputs_2d = pca.fit_transform(outputs)
                clustering_analysis["pca_2d"] = outputs_2d
                clustering_analysis["pca_explained_variance"] = pca.explained_variance_ratio_
                
                # t-SNE for non-linear visualization (only if we have sufficient samples)
                if len(outputs) > 30:
                    try:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(outputs)//4))
                        outputs_tsne = tsne.fit_transform(outputs)
                        clustering_analysis["tsne_2d"] = outputs_tsne
                    except Exception as e:
                        self.logger.warning(f"t-SNE failed: {e}")
                        
            except Exception as e:
                self.logger.warning(f"PCA failed: {e}")
        
        return clustering_analysis
    
    def _create_market_condition_grid(self) -> List[Dict[str, float]]:
        """Create a grid of different market conditions."""
        conditions = []
        
        # Vary key market parameters
        trends = [-0.1, -0.05, 0.0, 0.05, 0.1]
        volatilities = [0.01, 0.02, 0.03, 0.05]
        correlations = [0.1, 0.3, 0.5, 0.8]
        
        for trend in trends:
            for vol in volatilities:
                for corr in correlations:
                    conditions.append({
                        "trend": trend,
                        "volatility": vol,
                        "correlation": corr
                    })
        
        return conditions
    
    def _create_observation_from_condition(self, condition: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Create a synthetic observation based on market conditions."""
        # This is a simplified version - in practice, you'd want to create
        # more realistic synthetic data based on the conditions
        obs = self.train_env.reset()
        
        # Modify observation based on conditions
        tensor_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Add some noise based on volatility
                noise = np.random.normal(0, condition["volatility"], value.shape)
                modified_value = value + noise
                tensor_obs[key] = torch.FloatTensor(modified_value).unsqueeze(0).to(self.device)
            else:
                tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
        
        return tensor_obs
    
    def _cluster_decisions(self, decisions: np.ndarray) -> Dict[str, Any]:
        """Cluster decision patterns."""
        if len(decisions) < 3:
            return {"n_clusters": 0, "clusters": [], "warning": "Insufficient data for clustering"}
        
        # Check for data diversity
        unique_decisions = np.unique(decisions, axis=0)
        n_unique = len(unique_decisions)
        
        if n_unique < 2:
            return {
                "n_clusters": 1, 
                "clusters": np.zeros(len(decisions)),
                "centers": decisions[:1] if len(decisions) > 0 else np.array([]),
                "warning": "All decisions are identical"
            }
        
        # Adaptive number of clusters
        n_clusters = min(3, max(2, n_unique // 3))
        
        try:
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42,
                n_init=10,
                max_iter=300,
                tol=1e-4
            )
            clusters = kmeans.fit_predict(decisions)
            
            return {
                "n_clusters": n_clusters,
                "clusters": clusters,
                "centers": kmeans.cluster_centers_,
                "inertia": kmeans.inertia_
            }
        except Exception as e:
            self.logger.warning(f"Decision clustering failed: {e}")
            return {
                "n_clusters": 1,
                "clusters": np.zeros(len(decisions)),
                "centers": decisions[:1] if len(decisions) > 0 else np.array([]),
                "error": str(e)
            }
    
    def _analyze_network_evolution(self) -> Dict[str, Any]:
        """Analyze how the network has evolved from initialization."""
        evolution_analysis = {}
        
        # Compare current weights to what they might have been at initialization
        total_params = 0
        total_change = 0
        
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                # Estimate initial values (assuming normal initialization)
                fan_in = param.shape[1] if len(param.shape) > 1 else param.shape[0]
                std_init = np.sqrt(2.0 / fan_in)  # He initialization
                
                current_values = param.data.cpu().numpy()
                estimated_init = np.random.normal(0, std_init, current_values.shape)
                
                change = np.mean(np.abs(current_values - estimated_init))
                total_change += change * param.numel()
                total_params += param.numel()
        
        evolution_analysis["average_parameter_change"] = total_change / total_params if total_params > 0 else 0
        evolution_analysis["total_parameters"] = total_params
        
        return evolution_analysis
    
    def _generate_scenario_data(self, params: Dict[str, float]) -> np.ndarray:
        """Generate synthetic market data for a scenario."""
        n_steps = 100
        n_assets = self.train_env.n_assets
        
        # Generate correlated returns
        correlation_matrix = np.full((n_assets, n_assets), params["correlation"])
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate random returns
        returns = np.random.multivariate_normal(
            mean=[params["trend"]] * n_assets,
            cov=correlation_matrix * params["volatility"]**2,
            size=n_steps
        )
        
        # Convert to prices
        prices = np.cumprod(1 + returns, axis=0)
        
        return prices
    
    def _analyze_scenario_behavior(self, scenario_data: np.ndarray) -> Dict[str, Any]:
        """Analyze network behavior for a specific scenario."""
        actions = []
        
        # Create observations from scenario data
        for i in range(min(50, len(scenario_data) - self.train_env.window_size)):
            # Create a synthetic observation
            obs = self.train_env.reset()
            
            # Replace price data with scenario data
            if "ohlcv" in obs:
                window_data = scenario_data[i:i+self.train_env.window_size]
                # Simplified - just use closing prices
                obs["ohlcv"][:, :, 3] = window_data  # Assuming close price is index 3
            
            tensor_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _ = self.agent.get_intended_action(tensor_obs)
            
            actions.append(action)
        
        actions = np.array(actions)
        
        return {
            "mean_action": np.mean(actions, axis=0),
            "std_action": np.std(actions, axis=0),
            "action_range": np.ptp(actions, axis=0),
            "action_distribution": actions
        }
    
    def _compare_scenarios(self, scenario_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare behavior across different scenarios."""
        comparison = {}
        
        scenario_names = [name for name in scenario_analysis.keys() if name != "comparison"]
        
        # Compare mean actions
        mean_actions = {}
        for name in scenario_names:
            mean_actions[name] = scenario_analysis[name]["mean_action"]
        
        comparison["mean_action_comparison"] = mean_actions
        
        # Calculate action variability across scenarios
        all_means = np.array(list(mean_actions.values()))
        comparison["cross_scenario_variability"] = np.std(all_means, axis=0)
        
        return comparison
    
    def _flatten_observation(self, obs: Dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten observation dictionary to a single array."""
        flat_obs = []
        for tensor in obs.values():
            flat_obs.extend(tensor.flatten().cpu().numpy())
        return np.array(flat_obs)
    
    def plot_decision_boundaries(self, boundary_analysis: Dict[str, Any]) -> None:
        """Plot decision boundary analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Decision Boundary Analysis', fontsize=16)
        
        decisions = np.array(boundary_analysis["decisions"])
        conditions = boundary_analysis["conditions"]
        
        # Extract condition parameters
        trends = [c["trend"] for c in conditions]
        volatilities = [c["volatility"] for c in conditions]
        correlations = [c["correlation"] for c in conditions]
        
        # 1. Decision vs Trend
        ax1 = axes[0, 0]
        for i in range(min(3, decisions.shape[1])):  # Show first 3 assets
            ax1.scatter(trends, decisions[:, i], alpha=0.6, label=f'Asset {i+1}')
        ax1.set_xlabel('Market Trend')
        ax1.set_ylabel('Decision (Action)')
        ax1.set_title('Decisions vs Market Trend')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Decision vs Volatility
        ax2 = axes[0, 1]
        for i in range(min(3, decisions.shape[1])):
            ax2.scatter(volatilities, decisions[:, i], alpha=0.6, label=f'Asset {i+1}')
        ax2.set_xlabel('Market Volatility')
        ax2.set_ylabel('Decision (Action)')
        ax2.set_title('Decisions vs Market Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Decision vs Correlation
        ax3 = axes[1, 0]
        for i in range(min(3, decisions.shape[1])):
            ax3.scatter(correlations, decisions[:, i], alpha=0.6, label=f'Asset {i+1}')
        ax3.set_xlabel('Market Correlation')
        ax3.set_ylabel('Decision (Action)')
        ax3.set_title('Decisions vs Market Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Decision distribution
        ax4 = axes[1, 1]
        for i in range(min(3, decisions.shape[1])):
            ax4.hist(decisions[:, i], bins=20, alpha=0.6, label=f'Asset {i+1}')
        ax4.set_xlabel('Decision Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Decision Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'decision_boundaries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Decision boundaries saved to {self.output_dir / 'decision_boundaries.png'}")
    
    def plot_activation_patterns(self, activation_analysis: Dict[str, Any]) -> None:
        """Plot activation pattern analysis."""
        layer_stats = activation_analysis.get("layer_statistics", {})
        if not layer_stats:
            return
        
        n_layers = len(layer_stats)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Activation Pattern Analysis', fontsize=16)
        
        layer_names = list(layer_stats.keys())[:10]  # Limit to first 10 layers
        
        # 1. Mean activation by layer
        ax1 = axes[0, 0]
        mean_activations = [layer_stats[name]["mean_activation"] for name in layer_names]
        ax1.bar(range(len(layer_names)), mean_activations)
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
        ax1.set_ylabel('Mean Activation')
        ax1.set_title('Mean Activation by Layer')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sparsity by layer
        ax2 = axes[0, 1]
        sparsities = [layer_stats[name]["sparsity"] for name in layer_names]
        ax2.bar(range(len(layer_names)), sparsities, color='orange')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
        ax2.set_ylabel('Sparsity (Fraction of Zeros)')
        ax2.set_title('Activation Sparsity by Layer')
        ax2.grid(True, alpha=0.3)
        
        # 3. Activation distribution for first layer
        ax3 = axes[1, 0]
        if layer_names:
            first_layer_dist = layer_stats[layer_names[0]]["activation_distribution"]
            ax3.hist(first_layer_dist, bins=50, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Activation Value')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Activation Distribution: {layer_names[0].split(".")[-1]}')
            ax3.grid(True, alpha=0.3)
        
        # 4. Standard deviation by layer
        ax4 = axes[1, 1]
        std_activations = [layer_stats[name]["std_activation"] for name in layer_names]
        ax4.bar(range(len(layer_names)), std_activations, color='green')
        ax4.set_xticks(range(len(layer_names)))
        ax4.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
        ax4.set_ylabel('Activation Std')
        ax4.set_title('Activation Variability by Layer')
        ax4.grid(True, alpha=0.3)
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'activation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Activation patterns saved to {self.output_dir / 'activation_patterns.png'}")
    
    def plot_learning_progression(self, progression_analysis: Dict[str, Any]) -> None:
        """Plot learning progression analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Progression Analysis', fontsize=16)
        
        # 1. Sharpe ratio progression
        ax1 = axes[0, 0]
        if "sharpe_trend" in progression_analysis:
            sharpe_data = progression_analysis["sharpe_trend"]
            episodes = range(len(sharpe_data["values"]))
            ax1.plot(episodes, sharpe_data["values"], 'b-', alpha=0.7)
            
            # Add trend line
            if len(sharpe_data["values"]) > 1:
                z = np.polyfit(episodes, sharpe_data["values"], 1)
                p = np.poly1d(z)
                ax1.plot(episodes, p(episodes), "r--", alpha=0.8, 
                        label=f'Trend: {z[0]:.4f}')
                ax1.legend()
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Sharpe Ratio Progression')
        ax1.grid(True, alpha=0.3)
        
        # 2. Return progression
        ax2 = axes[0, 1]
        if "return_trend" in progression_analysis:
            return_data = progression_analysis["return_trend"]
            episodes = range(len(return_data["values"]))
            ax2.plot(episodes, return_data["values"], 'g-', alpha=0.7)
            
            # Add cumulative improvement annotation
            improvement = return_data.get("cumulative_improvement", 0)
            ax2.text(0.05, 0.95, f'Total Improvement: {improvement:.4f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Return')
        ax2.set_title('Return Progression')
        ax2.grid(True, alpha=0.3)
        
        # 3. Network evolution
        ax3 = axes[1, 0]
        if "network_evolution" in progression_analysis:
            evolution_data = progression_analysis["network_evolution"]
            param_change = evolution_data.get("average_parameter_change", 0)
            total_params = evolution_data.get("total_parameters", 0)
            
            ax3.bar(['Parameter Change'], [param_change], color='purple', alpha=0.7)
            ax3.set_ylabel('Average Change')
            ax3.set_title('Network Parameter Evolution')
            ax3.text(0, param_change + param_change*0.1, 
                    f'Total Params: {total_params:,}', ha='center')
        
        # 4. Learning rate analysis
        ax4 = axes[1, 1]
        if "return_trend" in progression_analysis:
            return_data = progression_analysis["return_trend"]
            learning_rate = return_data.get("learning_rate", 0)
            
            # Show learning rate over time (simplified)
            if len(return_data["values"]) > 1:
                returns = return_data["values"]
                learning_rates = np.diff(returns)
                episodes = range(1, len(returns))
                ax4.plot(episodes, learning_rates, 'r-', alpha=0.7)
                ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Episode-to-Episode Change')
                ax4.set_title('Learning Rate Over Time')
                ax4.grid(True, alpha=0.3)
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'learning_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Learning progression saved to {self.output_dir / 'learning_progression.png'}")
    
    def plot_scenario_comparison(self, scenario_analysis: Dict[str, Any]) -> None:
        """Plot scenario comparison analysis."""
        scenario_names = [name for name in scenario_analysis.keys() if name != "comparison"]
        if not scenario_names:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Market Scenario Analysis', fontsize=16)
        
        # 1. Mean actions by scenario
        ax1 = axes[0, 0]
        n_assets = len(scenario_analysis[scenario_names[0]]["mean_action"])
        x_pos = np.arange(len(scenario_names))
        width = 0.8 / n_assets
        
        for i in range(n_assets):
            means = [scenario_analysis[name]["mean_action"][i] for name in scenario_names]
            ax1.bar(x_pos + i * width, means, width, label=f'Asset {i+1}', alpha=0.7)
        
        ax1.set_xlabel('Market Scenario')
        ax1.set_ylabel('Mean Action')
        ax1.set_title('Mean Actions by Scenario')
        ax1.set_xticks(x_pos + width * (n_assets-1) / 2)
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Action variability by scenario
        ax2 = axes[0, 1]
        for i in range(n_assets):
            stds = [scenario_analysis[name]["std_action"][i] for name in scenario_names]
            ax2.bar(x_pos + i * width, stds, width, label=f'Asset {i+1}', alpha=0.7)
        
        ax2.set_xlabel('Market Scenario')
        ax2.set_ylabel('Action Std')
        ax2.set_title('Action Variability by Scenario')
        ax2.set_xticks(x_pos + width * (n_assets-1) / 2)
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-scenario variability
        ax3 = axes[1, 0]
        if "comparison" in scenario_analysis:
            cross_var = scenario_analysis["comparison"].get("cross_scenario_variability", [])
            if len(cross_var) > 0:
                ax3.bar(range(len(cross_var)), cross_var, alpha=0.7, color='red')
                ax3.set_xlabel('Asset')
                ax3.set_ylabel('Cross-Scenario Variability')
                ax3.set_title('Adaptability to Different Scenarios')
                ax3.set_xticks(range(len(cross_var)))
                ax3.set_xticklabels([f'Asset {i+1}' for i in range(len(cross_var))])
                ax3.grid(True, alpha=0.3)
        
        # 4. Action distribution comparison (first asset)
        ax4 = axes[1, 1]
        for name in scenario_names[:3]:  # Show first 3 scenarios
            actions = scenario_analysis[name]["action_distribution"]
            if len(actions) > 0:
                ax4.hist(actions[:, 0], bins=20, alpha=0.6, label=name)
        
        ax4.set_xlabel('Action Value (Asset 1)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Action Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Scenario comparison saved to {self.output_dir / 'scenario_comparison.png'}")
    
    def plot_behavior_clustering(self, clustering_analysis: Dict[str, Any]) -> None:
        """Plot behavior clustering analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Behavior Clustering', fontsize=16)
        
        # 1. PCA visualization
        ax1 = axes[0, 0]
        if "pca_2d" in clustering_analysis:
            pca_data = clustering_analysis["pca_2d"]
            clusters = clustering_analysis["clusters"]
            
            scatter = ax1.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, 
                                cmap='viridis', alpha=0.6)
            ax1.set_xlabel('First Principal Component')
            ax1.set_ylabel('Second Principal Component')
            ax1.set_title('PCA of Network Outputs')
            plt.colorbar(scatter, ax=ax1)
            
            # Add explained variance
            if "pca_explained_variance" in clustering_analysis:
                var_explained = clustering_analysis["pca_explained_variance"]
                ax1.text(0.05, 0.95, f'Explained Variance: {sum(var_explained):.2%}', 
                        transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        else:
            ax1.text(0.5, 0.5, 'PCA data not available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('PCA of Network Outputs (Not Available)')
        
        # 2. t-SNE visualization
        ax2 = axes[0, 1]
        if "tsne_2d" in clustering_analysis:
            tsne_data = clustering_analysis["tsne_2d"]
            clusters = clustering_analysis["clusters"]
            
            scatter = ax2.scatter(tsne_data[:, 0], tsne_data[:, 1], c=clusters, 
                                cmap='viridis', alpha=0.6)
            ax2.set_xlabel('t-SNE Dimension 1')
            ax2.set_ylabel('t-SNE Dimension 2')
            ax2.set_title('t-SNE of Network Outputs')
            plt.colorbar(scatter, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 't-SNE data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('t-SNE of Network Outputs (Not Available)')
        
        # 3. Cluster sizes
        ax3 = axes[1, 0]
        cluster_analysis = clustering_analysis.get("cluster_analysis", {})
        if cluster_analysis:
            cluster_ids = list(cluster_analysis.keys())
            cluster_sizes = [cluster_analysis[i]["size"] for i in cluster_ids]
            
            if cluster_sizes:  # Only plot if we have data
                ax3.bar(cluster_ids, cluster_sizes, alpha=0.7, color='skyblue')
                ax3.set_xlabel('Cluster ID')
                ax3.set_ylabel('Cluster Size')
                ax3.set_title('Cluster Size Distribution')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No cluster data available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cluster Size Distribution (No Data)')
        else:
            ax3.text(0.5, 0.5, 'Cluster analysis not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cluster Size Distribution (Not Available)')
        
        # 4. Cluster characteristics (mean output)
        ax4 = axes[1, 1]
        if cluster_analysis:
            n_clusters = len(cluster_analysis)
            
            # Check if we have valid output means
            valid_clusters = []
            for i in range(n_clusters):
                if i in cluster_analysis and len(cluster_analysis[i].get("output_mean", [])) > 0:
                    valid_clusters.append(i)
            
            if valid_clusters:
                output_dims = len(cluster_analysis[valid_clusters[0]]["output_mean"])
                
                # Show first few output dimensions
                for dim in range(min(3, output_dims)):
                    means = []
                    cluster_ids = []
                    for i in valid_clusters:
                        output_mean = cluster_analysis[i]["output_mean"]
                        if len(output_mean) > dim and not np.isnan(output_mean[dim]):
                            means.append(output_mean[dim])
                            cluster_ids.append(i)
                    
                    if means:  # Only plot if we have valid data
                        ax4.plot(cluster_ids, means, marker='o', label=f'Output Dim {dim+1}')
                
                if ax4.get_lines():  # Only set labels if we plotted something
                    ax4.set_xlabel('Cluster ID')
                    ax4.set_ylabel('Mean Output Value')
                    ax4.set_title('Cluster Output Characteristics')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No valid cluster characteristics', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Cluster Output Characteristics (No Valid Data)')
            else:
                ax4.text(0.5, 0.5, 'No valid clusters found', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Cluster Output Characteristics (No Valid Clusters)')
        else:
            ax4.text(0.5, 0.5, 'Cluster analysis not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cluster Output Characteristics (Not Available)')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'behavior_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Behavior clustering saved to {self.output_dir / 'behavior_clustering.png'}")
    
    def create_interactive_dashboard(self, results: Dict[str, Any]) -> None:
        """Create an interactive dashboard using Plotly."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Decision Boundaries', 'Activation Patterns', 
                              'Learning Progression', 'Scenario Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Decision boundaries (if available)
            if "decision_boundaries" in results:
                boundary_data = results["decision_boundaries"]
                decisions = np.array(boundary_data["decisions"])
                conditions = boundary_data["conditions"]
                trends = [c["trend"] for c in conditions]
                
                if len(decisions) > 0:
                    fig.add_trace(
                        go.Scatter(x=trends, y=decisions[:, 0], mode='markers',
                                 name='Asset 1 Decisions', opacity=0.6),
                        row=1, col=1
                    )
            
            # 2. Activation patterns (if available)
            if "activation_patterns" in results:
                layer_stats = results["activation_patterns"].get("layer_statistics", {})
                if layer_stats:
                    layer_names = list(layer_stats.keys())[:5]
                    mean_activations = [layer_stats[name]["mean_activation"] for name in layer_names]
                    
                    fig.add_trace(
                        go.Bar(x=[name.split('.')[-1] for name in layer_names], 
                              y=mean_activations, name='Mean Activation'),
                        row=1, col=2
                    )
            
            # 3. Learning progression (if available)
            if "learning_progression" in results:
                if "sharpe_trend" in results["learning_progression"]:
                    sharpe_data = results["learning_progression"]["sharpe_trend"]
                    episodes = list(range(len(sharpe_data["values"])))
                    
                    fig.add_trace(
                        go.Scatter(x=episodes, y=sharpe_data["values"], 
                                 mode='lines', name='Sharpe Ratio'),
                        row=2, col=1
                    )
            
            # 4. Scenario comparison (if available)
            if "scenario_analysis" in results:
                scenario_names = [name for name in results["scenario_analysis"].keys() 
                                if name != "comparison"]
                if scenario_names:
                    mean_actions = [results["scenario_analysis"][name]["mean_action"][0] 
                                  for name in scenario_names]
                    
                    fig.add_trace(
                        go.Bar(x=scenario_names, y=mean_actions, 
                              name='Mean Action (Asset 1)'),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title_text="Interactive Network Analysis Dashboard",
                showlegend=True,
                height=800
            )
            
            # Save interactive plot
            fig.write_html(str(self.output_dir / 'interactive_dashboard.html'))
            
            self.logger.info(f"Interactive dashboard saved to {self.output_dir / 'interactive_dashboard.html'}")
            
        except Exception as e:
            self.logger.warning(f"Could not create interactive dashboard: {e}")
            # Plotly might not be available or there might be other issues 