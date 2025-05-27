"""
Sensitivity Analyzer

This module provides tools for analyzing the sensitivity of trained neural networks
to input perturbations, helping understand what features the network has learned
to focus on and how robust the network is to input variations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.agents.base_agent import BaseAgent
from environments.trading_env import TradingEnv


class SensitivityAnalyzer:
    """
    Analyzes network sensitivity to input perturbations.
    
    Provides tools for understanding which input features are most important
    to the network's decisions and how robust the network is to noise.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        train_env: TradingEnv,
        val_env: TradingEnv,
        output_dir: Path,
        logger
    ):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            agent: Trained agent with neural network
            train_env: Training environment
            val_env: Validation environment
            output_dir: Directory to save visualizations
            logger: Logger instance
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.output_dir = output_dir
        self.logger = logger
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get network and device
        self.network = agent.get_model()
        self.device = next(self.network.parameters()).device
        
        # Set network to evaluation mode
        self.network.eval()
    
    def analyze_full_sensitivity(
        self,
        n_samples: int = 1000,
        perturbation_scales: List[float] = [0.01, 0.05, 0.1, 0.2],
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete sensitivity analysis.
        
        Args:
            n_samples: Number of samples to use for analysis
            perturbation_scales: List of perturbation scales to test
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        results = {}
        
        # Get sample data
        sample_data = self._get_sample_data(n_samples)
        
        # Analyze input sensitivity
        results["input_sensitivity"] = self.analyze_input_sensitivity(
            sample_data, perturbation_scales
        )
        
        # Analyze feature importance
        results["feature_importance"] = self.analyze_feature_importance(sample_data)
        
        # Analyze output distributions
        results["output_distributions"] = self.analyze_output_distributions(
            sample_data, perturbation_scales
        )
        
        # Analyze gradient-based sensitivity
        results["gradient_sensitivity"] = self.analyze_gradient_sensitivity(sample_data)
        
        if save_plots:
            self.plot_sensitivity_analysis(results)
            self.plot_feature_importance(results["feature_importance"])
            self.plot_output_distributions(results["output_distributions"])
            self.plot_gradient_analysis(results["gradient_sensitivity"])
        
        return results
    
    def _get_sample_data(self, n_samples: int) -> List[Dict[str, torch.Tensor]]:
        """Get sample observations from both environments."""
        samples = []
        
        # Get samples from training environment
        train_samples = min(n_samples // 2, 500)
        for _ in range(train_samples):
            obs = self.train_env.reset()
            # Convert to tensors
            tensor_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
            samples.append(tensor_obs)
        
        # Get samples from validation environment
        val_samples = n_samples - train_samples
        for _ in range(val_samples):
            obs = self.val_env.reset()
            # Convert to tensors
            tensor_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
            samples.append(tensor_obs)
        
        return samples
    
    def analyze_input_sensitivity(
        self,
        sample_data: List[Dict[str, torch.Tensor]],
        perturbation_scales: List[float]
    ) -> Dict[str, Any]:
        """Analyze how sensitive the network is to input perturbations."""
        sensitivity_results = {
            "perturbation_scales": perturbation_scales,
            "output_changes": {},
            "robustness_metrics": {}
        }
        
        for scale in perturbation_scales:
            output_changes = []
            
            for sample in sample_data[:100]:  # Use subset for efficiency
                # Get original output
                with torch.no_grad():
                    original_output = self.network(sample)
                
                # Create perturbed versions
                perturbed_outputs = []
                for _ in range(10):  # Multiple perturbations per sample
                    perturbed_sample = self._perturb_observation(sample, scale)
                    with torch.no_grad():
                        perturbed_output = self.network(perturbed_sample)
                    perturbed_outputs.append(perturbed_output)
                
                # Calculate output changes
                changes = []
                for perturbed_output in perturbed_outputs:
                    change = self._calculate_output_change(original_output, perturbed_output)
                    changes.append(change)
                
                output_changes.extend(changes)
            
            sensitivity_results["output_changes"][scale] = {
                "mean": np.mean(output_changes),
                "std": np.std(output_changes),
                "median": np.median(output_changes),
                "percentiles": {
                    "25": np.percentile(output_changes, 25),
                    "75": np.percentile(output_changes, 75),
                    "95": np.percentile(output_changes, 95)
                }
            }
            
            # Calculate robustness metrics
            sensitivity_results["robustness_metrics"][scale] = {
                "stability_ratio": np.mean(np.array(output_changes) < 0.1),
                "max_change": np.max(output_changes),
                "change_variance": np.var(output_changes)
            }
        
        return sensitivity_results
    
    def analyze_feature_importance(
        self,
        sample_data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Analyze which input features are most important for the network's decisions."""
        feature_importance = {}
        
        for sample in sample_data[:50]:  # Use subset for efficiency
            # Get original output
            with torch.no_grad():
                original_output = self.network(sample)
            
            # Test importance of each input component
            for input_key, input_tensor in sample.items():
                if input_tensor.numel() > 1:  # Skip scalar inputs
                    importance_scores = []
                    
                    # Test each feature dimension
                    for dim in range(input_tensor.shape[-1]):
                        # Zero out this dimension
                        modified_sample = {k: v.clone() for k, v in sample.items()}
                        if len(input_tensor.shape) == 3:  # Time series data
                            modified_sample[input_key][:, :, dim] = 0
                        elif len(input_tensor.shape) == 2:  # Vector data
                            modified_sample[input_key][:, dim] = 0
                        
                        # Get output with this feature removed
                        with torch.no_grad():
                            modified_output = self.network(modified_sample)
                        
                        # Calculate importance as output change
                        importance = self._calculate_output_change(original_output, modified_output)
                        importance_scores.append(importance)
                    
                    if input_key not in feature_importance:
                        feature_importance[input_key] = []
                    feature_importance[input_key].append(importance_scores)
        
        # Aggregate importance scores
        aggregated_importance = {}
        for input_key, importance_lists in feature_importance.items():
            # Average across samples
            importance_array = np.array(importance_lists)
            aggregated_importance[input_key] = {
                "mean_importance": np.mean(importance_array, axis=0),
                "std_importance": np.std(importance_array, axis=0),
                "max_importance": np.max(importance_array, axis=0),
                "feature_ranking": np.argsort(np.mean(importance_array, axis=0))[::-1]
            }
        
        return aggregated_importance
    
    def analyze_output_distributions(
        self,
        sample_data: List[Dict[str, torch.Tensor]],
        perturbation_scales: List[float]
    ) -> Dict[str, Any]:
        """Analyze how output distributions change with input perturbations."""
        distribution_analysis = {}
        
        # Get baseline outputs
        baseline_outputs = []
        for sample in sample_data:
            with torch.no_grad():
                output = self.network(sample)
                baseline_outputs.append(self._extract_output_values(output))
        
        baseline_outputs = np.array(baseline_outputs)
        distribution_analysis["baseline"] = {
            "mean": np.mean(baseline_outputs, axis=0),
            "std": np.std(baseline_outputs, axis=0),
            "distribution": baseline_outputs
        }
        
        # Analyze perturbed outputs
        for scale in perturbation_scales:
            perturbed_outputs = []
            
            for sample in sample_data:
                perturbed_sample = self._perturb_observation(sample, scale)
                with torch.no_grad():
                    output = self.network(perturbed_sample)
                    perturbed_outputs.append(self._extract_output_values(output))
            
            perturbed_outputs = np.array(perturbed_outputs)
            distribution_analysis[f"perturbed_{scale}"] = {
                "mean": np.mean(perturbed_outputs, axis=0),
                "std": np.std(perturbed_outputs, axis=0),
                "distribution": perturbed_outputs,
                "ks_test": [
                    stats.ks_2samp(baseline_outputs[:, i], perturbed_outputs[:, i])
                    for i in range(baseline_outputs.shape[1])
                ]
            }
        
        return distribution_analysis
    
    def analyze_gradient_sensitivity(
        self,
        sample_data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Analyze gradient-based sensitivity measures."""
        gradient_analysis = {}
        
        # Enable gradients for input tensors
        for sample in sample_data[:20]:  # Use subset for efficiency
            for key, tensor in sample.items():
                tensor.requires_grad_(True)
            
            # Forward pass
            output = self.network(sample)
            
            # Calculate gradients for each output component
            if isinstance(output, dict):
                for output_key, output_tensor in output.items():
                    if output_tensor.numel() == 1:
                        # Scalar output
                        output_tensor.backward(retain_graph=True)
                    else:
                        # Vector output - use sum for simplicity
                        output_tensor.sum().backward(retain_graph=True)
                    
                    # Collect gradients
                    gradients = {}
                    for input_key, input_tensor in sample.items():
                        if input_tensor.grad is not None:
                            gradients[input_key] = input_tensor.grad.clone().detach()
                    
                    if output_key not in gradient_analysis:
                        gradient_analysis[output_key] = []
                    gradient_analysis[output_key].append(gradients)
                    
                    # Clear gradients
                    for input_tensor in sample.values():
                        if input_tensor.grad is not None:
                            input_tensor.grad.zero_()
            
            # Clear requires_grad
            for tensor in sample.values():
                tensor.requires_grad_(False)
        
        # Aggregate gradient statistics
        aggregated_gradients = {}
        for output_key, gradient_list in gradient_analysis.items():
            aggregated_gradients[output_key] = {}
            
            for input_key in gradient_list[0].keys():
                gradients = [g[input_key] for g in gradient_list if input_key in g]
                if gradients:
                    gradient_tensor = torch.stack(gradients)
                    aggregated_gradients[output_key][input_key] = {
                        "mean_abs_gradient": torch.mean(torch.abs(gradient_tensor), dim=0),
                        "std_gradient": torch.std(gradient_tensor, dim=0),
                        "max_gradient": torch.max(torch.abs(gradient_tensor), dim=0)[0]
                    }
        
        return aggregated_gradients
    
    def _perturb_observation(
        self,
        observation: Dict[str, torch.Tensor],
        scale: float
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to an observation."""
        perturbed_obs = {}
        for key, tensor in observation.items():
            noise = torch.randn_like(tensor) * scale
            perturbed_obs[key] = tensor + noise
        return perturbed_obs
    
    def _calculate_output_change(
        self,
        original_output: Dict[str, torch.Tensor],
        perturbed_output: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate the magnitude of change between two outputs."""
        total_change = 0.0
        count = 0
        
        for key in original_output.keys():
            if key in perturbed_output:
                orig = original_output[key].flatten()
                pert = perturbed_output[key].flatten()
                change = torch.norm(orig - pert, p=2).item()
                total_change += change
                count += 1
        
        return total_change / count if count > 0 else 0.0
    
    def _extract_output_values(self, output: Dict[str, torch.Tensor]) -> np.ndarray:
        """Extract numerical values from network output."""
        values = []
        for tensor in output.values():
            values.extend(tensor.flatten().cpu().numpy())
        return np.array(values)
    
    def plot_sensitivity_analysis(self, results: Dict[str, Any]) -> None:
        """Plot sensitivity analysis results."""
        sensitivity_data = results["input_sensitivity"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Input Sensitivity Analysis', fontsize=16)
        
        scales = sensitivity_data["perturbation_scales"]
        
        # 1. Mean output change vs perturbation scale
        ax1 = axes[0, 0]
        mean_changes = [sensitivity_data["output_changes"][scale]["mean"] for scale in scales]
        std_changes = [sensitivity_data["output_changes"][scale]["std"] for scale in scales]
        
        ax1.errorbar(scales, mean_changes, yerr=std_changes, marker='o', capsize=5)
        ax1.set_xlabel('Perturbation Scale')
        ax1.set_ylabel('Mean Output Change')
        ax1.set_title('Sensitivity to Input Perturbations')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # 2. Robustness metrics
        ax2 = axes[0, 1]
        stability_ratios = [sensitivity_data["robustness_metrics"][scale]["stability_ratio"] for scale in scales]
        ax2.plot(scales, stability_ratios, marker='s', color='green')
        ax2.set_xlabel('Perturbation Scale')
        ax2.set_ylabel('Stability Ratio')
        ax2.set_title('Network Stability')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 3. Distribution of output changes
        ax3 = axes[1, 0]
        for i, scale in enumerate(scales[:3]):  # Show first 3 scales
            changes = []
            # Reconstruct changes from statistics (approximation)
            stats = sensitivity_data["output_changes"][scale]
            changes = np.random.normal(stats["mean"], stats["std"], 100)
            ax3.hist(changes, bins=20, alpha=0.6, label=f'Scale {scale}')
        
        ax3.set_xlabel('Output Change Magnitude')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Output Changes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Max change vs scale
        ax4 = axes[1, 1]
        max_changes = [sensitivity_data["robustness_metrics"][scale]["max_change"] for scale in scales]
        ax4.plot(scales, max_changes, marker='^', color='red')
        ax4.set_xlabel('Perturbation Scale')
        ax4.set_ylabel('Maximum Output Change')
        ax4.set_title('Worst-Case Sensitivity')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sensitivity analysis saved to {self.output_dir / 'sensitivity_analysis.png'}")
    
    def plot_feature_importance(self, feature_importance: Dict[str, Any]) -> None:
        """Plot feature importance analysis."""
        if not feature_importance:
            return
        
        fig, axes = plt.subplots(len(feature_importance), 1, 
                                figsize=(12, 4 * len(feature_importance)))
        if len(feature_importance) == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Analysis', fontsize=16)
        
        for i, (input_key, importance_data) in enumerate(feature_importance.items()):
            ax = axes[i]
            
            mean_importance = importance_data["mean_importance"]
            std_importance = importance_data["std_importance"]
            
            x_pos = range(len(mean_importance))
            ax.bar(x_pos, mean_importance, yerr=std_importance, alpha=0.7, capsize=3)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance Score')
            ax.set_title(f'Feature Importance: {input_key}')
            ax.grid(True, alpha=0.3)
            
            # Highlight top features
            ranking = importance_data["feature_ranking"][:5]  # Top 5
            for rank in ranking:
                ax.axvline(x=rank, color='red', linestyle='--', alpha=0.5)
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance saved to {self.output_dir / 'feature_importance.png'}")
    
    def plot_output_distributions(self, distribution_analysis: Dict[str, Any]) -> None:
        """Plot output distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Output Distribution Analysis', fontsize=16)
        
        baseline = distribution_analysis["baseline"]["distribution"]
        
        # 1. Baseline distribution
        ax1 = axes[0, 0]
        if baseline.shape[1] > 1:
            for i in range(min(3, baseline.shape[1])):  # Show first 3 outputs
                ax1.hist(baseline[:, i], bins=30, alpha=0.6, label=f'Output {i}')
            ax1.legend()
        else:
            ax1.hist(baseline.flatten(), bins=30, alpha=0.7)
        ax1.set_xlabel('Output Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Baseline Output Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution shift with perturbations
        ax2 = axes[0, 1]
        perturbation_keys = [k for k in distribution_analysis.keys() if k.startswith('perturbed_')]
        
        for key in perturbation_keys[:3]:  # Show first 3 perturbation levels
            perturbed = distribution_analysis[key]["distribution"]
            scale = key.split('_')[1]
            if perturbed.shape[1] > 0:
                ax2.hist(perturbed[:, 0], bins=30, alpha=0.6, label=f'Scale {scale}')
        
        ax2.hist(baseline[:, 0], bins=30, alpha=0.6, label='Baseline', color='black')
        ax2.set_xlabel('Output Value (First Component)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution Shift with Perturbations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. KS test statistics
        ax3 = axes[1, 0]
        ks_stats = []
        scales = []
        
        for key in perturbation_keys:
            scale = float(key.split('_')[1])
            scales.append(scale)
            ks_tests = distribution_analysis[key]["ks_test"]
            # Average KS statistic across output components
            avg_ks = np.mean([test.statistic for test in ks_tests])
            ks_stats.append(avg_ks)
        
        if ks_stats:
            ax3.plot(scales, ks_stats, marker='o')
            ax3.set_xlabel('Perturbation Scale')
            ax3.set_ylabel('Average KS Statistic')
            ax3.set_title('Distribution Divergence')
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')
        
        # 4. Mean and std changes
        ax4 = axes[1, 1]
        baseline_mean = distribution_analysis["baseline"]["mean"]
        baseline_std = distribution_analysis["baseline"]["std"]
        
        mean_changes = []
        std_changes = []
        
        for key in perturbation_keys:
            perturbed_mean = distribution_analysis[key]["mean"]
            perturbed_std = distribution_analysis[key]["std"]
            
            mean_change = np.mean(np.abs(perturbed_mean - baseline_mean))
            std_change = np.mean(np.abs(perturbed_std - baseline_std))
            
            mean_changes.append(mean_change)
            std_changes.append(std_change)
        
        if mean_changes:
            ax4.plot(scales, mean_changes, marker='o', label='Mean Change')
            ax4.plot(scales, std_changes, marker='s', label='Std Change')
            ax4.set_xlabel('Perturbation Scale')
            ax4.set_ylabel('Change Magnitude')
            ax4.set_title('Statistical Moment Changes')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'output_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Output distributions saved to {self.output_dir / 'output_distributions.png'}")
    
    def plot_gradient_analysis(self, gradient_analysis: Dict[str, Any]) -> None:
        """Plot gradient-based sensitivity analysis."""
        if not gradient_analysis:
            return
        
        n_outputs = len(gradient_analysis)
        fig, axes = plt.subplots(n_outputs, 2, figsize=(15, 5 * n_outputs))
        if n_outputs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Gradient-Based Sensitivity Analysis', fontsize=16)
        
        for i, (output_key, gradient_data) in enumerate(gradient_analysis.items()):
            # 1. Mean absolute gradients
            ax1 = axes[i, 0]
            input_keys = list(gradient_data.keys())
            
            for j, input_key in enumerate(input_keys):
                grad_data = gradient_data[input_key]
                mean_abs_grad = grad_data["mean_abs_gradient"]
                
                if mean_abs_grad.numel() > 1:
                    # Plot gradient magnitude across features
                    ax1.plot(mean_abs_grad.cpu().numpy().flatten(), 
                            label=input_key, alpha=0.7)
                else:
                    # Single value
                    ax1.bar([j], [mean_abs_grad.item()], alpha=0.7, label=input_key)
            
            ax1.set_xlabel('Feature Index' if mean_abs_grad.numel() > 1 else 'Input')
            ax1.set_ylabel('Mean Absolute Gradient')
            ax1.set_title(f'Gradient Magnitude: {output_key}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Gradient statistics
            ax2 = axes[i, 1]
            max_grads = []
            std_grads = []
            
            for input_key in input_keys:
                grad_data = gradient_data[input_key]
                max_grad = grad_data["max_gradient"].max().item()
                std_grad = grad_data["std_gradient"].mean().item()
                
                max_grads.append(max_grad)
                std_grads.append(std_grad)
            
            x_pos = range(len(input_keys))
            width = 0.35
            
            ax2.bar([x - width/2 for x in x_pos], max_grads, width, 
                   label='Max Gradient', alpha=0.7)
            ax2.bar([x + width/2 for x in x_pos], std_grads, width, 
                   label='Std Gradient', alpha=0.7)
            
            ax2.set_xlabel('Input Type')
            ax2.set_ylabel('Gradient Value')
            ax2.set_title(f'Gradient Statistics: {output_key}')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(input_keys, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gradient analysis saved to {self.output_dir / 'gradient_analysis.png'}") 