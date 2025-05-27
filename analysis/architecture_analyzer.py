"""
Architecture Analyzer

This module provides tools for visualizing and analyzing the architecture
of trained neural networks, including layer dimensions, parameter counts,
network topology, and convolutional operation analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict
import warnings

from models.agents.base_agent import BaseAgent
from environments.trading_env import TradingEnv


class ArchitectureAnalyzer:
    """
    Analyzes and visualizes neural network architecture.
    
    Provides tools for understanding the structure, dimensions,
    and parameter distribution of trained networks.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        train_env: TradingEnv,
        output_dir: Path,
        logger
    ):
        """
        Initialize the architecture analyzer.
        
        Args:
            agent: Trained agent with neural network
            train_env: Training environment for getting sample inputs
            output_dir: Directory to save visualizations
            logger: Logger instance
        """
        self.agent = agent
        self.train_env = train_env
        self.output_dir = output_dir
        self.logger = logger
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get network and sample input
        self.network = agent.get_model()
        self.device = next(self.network.parameters()).device
        
        # Get sample observation for analysis
        self.sample_obs = self._get_sample_observation()
        
    def _get_sample_observation(self) -> Dict[str, torch.Tensor]:
        """Get a sample observation from the environment."""
        obs = self.train_env.reset()
        
        # Convert to tensors and move to device
        tensor_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            else:
                tensor_obs[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
        
        return tensor_obs
    
    def analyze_full_architecture(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform complete architecture analysis.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Analyze network structure
        results["structure"] = self.analyze_network_structure()
        
        # Analyze layer dimensions
        results["dimensions"] = self.analyze_layer_dimensions()
        
        # Analyze parameter distribution
        results["parameters"] = self.analyze_parameter_distribution()
        
        # Analyze data flow
        results["data_flow"] = self.analyze_data_flow()
        
        # Analyze Conv1D operations (price processor)
        conv1d_results = self.analyze_conv1d_operations(save_plots=save_plots)
        if conv1d_results:
            results["conv1d_analysis"] = conv1d_results
        
        # Analyze Conv2D operations (OHLCV processor)
        conv2d_results = self.analyze_conv2d_operations(save_plots=save_plots)
        if conv2d_results:
            results["conv2d_analysis"] = conv2d_results
        
        # Export to ONNX and create topology visualization
        onnx_results = self.analyze_onnx_export(save_plots=save_plots)
        if onnx_results:
            results["onnx_analysis"] = onnx_results
        
        if save_plots:
            self.plot_architecture_summary(results)
            self.plot_network_topology()
            self.plot_parameter_distribution(results["parameters"])
            self.plot_layer_dimensions(results["dimensions"])
            
        return results
    
    def analyze_network_structure(self) -> Dict[str, Any]:
        """Analyze the overall network structure."""
        structure = {
            "total_parameters": sum(p.numel() for p in self.network.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.network.parameters() if p.requires_grad),
            "layers": {},
            "modules": {}
        }
        
        # Analyze each module
        for name, module in self.network.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_info = {
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
                }
                
                # Add specific info for different layer types
                if isinstance(module, nn.Linear):
                    module_info["input_features"] = module.in_features
                    module_info["output_features"] = module.out_features
                elif isinstance(module, nn.Conv1d):
                    module_info["in_channels"] = module.in_channels
                    module_info["out_channels"] = module.out_channels
                    module_info["kernel_size"] = module.kernel_size
                elif isinstance(module, nn.LSTM):
                    module_info["input_size"] = module.input_size
                    module_info["hidden_size"] = module.hidden_size
                    module_info["num_layers"] = module.num_layers
                
                structure["modules"][name] = module_info
        
        return structure
    
    def analyze_layer_dimensions(self) -> Dict[str, Any]:
        """Analyze input/output dimensions of each layer."""
        dimensions = {}
        
        # Hook to capture layer outputs
        layer_outputs = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = {
                        "input_shape": input[0].shape if input and isinstance(input[0], torch.Tensor) else None,
                        "output_shape": output.shape,
                        "output_mean": output.mean().item(),
                        "output_std": output.std().item(),
                        "output_min": output.min().item(),
                        "output_max": output.max().item()
                    }
                elif isinstance(output, dict):
                    # Handle dictionary outputs (like from unified network)
                    layer_outputs[name] = {}
                    for key, tensor in output.items():
                        if isinstance(tensor, torch.Tensor):
                            layer_outputs[name][key] = {
                                "output_shape": tensor.shape,
                                "output_mean": tensor.mean().item(),
                                "output_std": tensor.std().item(),
                                "output_min": tensor.min().item(),
                                "output_max": tensor.max().item()
                            }
            return hook
        
        # Register hooks
        for name, module in self.network.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to capture dimensions
        with torch.no_grad():
            _ = self.network(self.sample_obs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        dimensions["layer_outputs"] = layer_outputs
        return dimensions
    
    def analyze_parameter_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of parameters across the network."""
        param_stats = {}
        
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy().flatten()
                param_stats[name] = {
                    "shape": list(param.shape),
                    "total_params": param.numel(),
                    "mean": float(np.mean(param_data)),
                    "std": float(np.std(param_data)),
                    "min": float(np.min(param_data)),
                    "max": float(np.max(param_data)),
                    "zero_fraction": float(np.mean(param_data == 0)),
                    "percentiles": {
                        "5": float(np.percentile(param_data, 5)),
                        "25": float(np.percentile(param_data, 25)),
                        "50": float(np.percentile(param_data, 50)),
                        "75": float(np.percentile(param_data, 75)),
                        "95": float(np.percentile(param_data, 95))
                    }
                }
        
        return param_stats
    
    def analyze_data_flow(self) -> Dict[str, Any]:
        """Analyze how data flows through the network."""
        data_flow = {
            "input_processors": [],
            "backbone_layers": [],
            "output_heads": []
        }
        
        # Analyze processors
        if hasattr(self.network, 'processors'):
            for name, processor in self.network.processors.items():
                if processor is not None:
                    data_flow["input_processors"].append({
                        "name": name,
                        "type": type(processor).__name__,
                        "output_dim": getattr(processor, 'get_output_dim', lambda: 'unknown')()
                    })
        
        # Analyze backbone
        if hasattr(self.network, 'backbone'):
            backbone_info = {
                "type": type(self.network.backbone).__name__,
                "input_dim": getattr(self.network.backbone, 'input_dim', 'unknown'),
                "output_dim": getattr(self.network.backbone, 'get_output_dim', lambda: 'unknown')()
            }
            data_flow["backbone_layers"].append(backbone_info)
        
        # Analyze heads
        if hasattr(self.network, 'heads'):
            for name, head in self.network.heads.items():
                if head is not None:
                    data_flow["output_heads"].append({
                        "name": name,
                        "type": type(head).__name__
                    })
        
        return data_flow
    
    def plot_architecture_summary(self, results: Dict[str, Any]) -> None:
        """Plot a summary of the network architecture."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Architecture Summary', fontsize=16)
        
        # 1. Parameter distribution by module type
        ax1 = axes[0, 0]
        module_types = defaultdict(int)
        for module_info in results["structure"]["modules"].values():
            module_types[module_info["type"]] += module_info["parameters"]
        
        if module_types:
            ax1.pie(module_types.values(), labels=module_types.keys(), autopct='%1.1f%%')
            ax1.set_title('Parameters by Module Type')
        
        # 2. Layer parameter counts
        ax2 = axes[0, 1]
        module_names = list(results["structure"]["modules"].keys())[:10]  # Top 10
        param_counts = [results["structure"]["modules"][name]["parameters"] for name in module_names]
        
        if param_counts:
            ax2.barh(range(len(module_names)), param_counts)
            ax2.set_yticks(range(len(module_names)))
            ax2.set_yticklabels([name.split('.')[-1] for name in module_names])
            ax2.set_xlabel('Parameter Count')
            ax2.set_title('Top Layers by Parameter Count')
        
        # 3. Data flow diagram
        ax3 = axes[1, 0]
        flow_info = results["data_flow"]
        
        # Create a simple flow diagram
        y_pos = 0.8
        ax3.text(0.1, y_pos, "Input Processors:", fontweight='bold', fontsize=12)
        y_pos -= 0.1
        for processor in flow_info["input_processors"]:
            ax3.text(0.2, y_pos, f"• {processor['name']} ({processor['type']})", fontsize=10)
            y_pos -= 0.08
        
        y_pos -= 0.05
        ax3.text(0.1, y_pos, "Backbone:", fontweight='bold', fontsize=12)
        y_pos -= 0.1
        for backbone in flow_info["backbone_layers"]:
            ax3.text(0.2, y_pos, f"• {backbone['type']}", fontsize=10)
            y_pos -= 0.08
        
        y_pos -= 0.05
        ax3.text(0.1, y_pos, "Output Heads:", fontweight='bold', fontsize=12)
        y_pos -= 0.1
        for head in flow_info["output_heads"]:
            ax3.text(0.2, y_pos, f"• {head['name']} ({head['type']})", fontsize=10)
            y_pos -= 0.08
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Network Data Flow')
        
        # 4. Network statistics
        ax4 = axes[1, 1]
        stats_text = [
            f"Total Parameters: {results['structure']['total_parameters']:,}",
            f"Trainable Parameters: {results['structure']['trainable_parameters']:,}",
            f"Total Modules: {len(results['structure']['modules'])}",
            f"Input Processors: {len(flow_info['input_processors'])}",
            f"Output Heads: {len(flow_info['output_heads'])}"
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(stats_text), fontsize=12, verticalalignment='top')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Network Statistics')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'architecture_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Architecture summary saved to {self.output_dir / 'architecture_summary.png'}")
    
    def plot_network_topology(self) -> None:
        """Plot the network topology as a graph."""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each module
        pos = {}
        y_positions = {"processors": 0.8, "backbone": 0.5, "heads": 0.2}
        x_spacing = {"processors": 0.2, "backbone": 0.5, "heads": 0.2}
        
        # Add processor nodes
        if hasattr(self.network, 'processors'):
            processor_names = [name for name, proc in self.network.processors.items() if proc is not None]
            for i, name in enumerate(processor_names):
                x_pos = 0.1 + i * x_spacing["processors"]
                pos[f"proc_{name}"] = (x_pos, y_positions["processors"])
                G.add_node(f"proc_{name}", type="processor", label=name)
        
        # Add backbone node
        if hasattr(self.network, 'backbone'):
            pos["backbone"] = (0.5, y_positions["backbone"])
            G.add_node("backbone", type="backbone", label="Backbone")
        
        # Add head nodes
        if hasattr(self.network, 'heads'):
            head_names = [name for name, head in self.network.heads.items() if head is not None]
            for i, name in enumerate(head_names):
                x_pos = 0.3 + i * x_spacing["heads"]
                pos[f"head_{name}"] = (x_pos, y_positions["heads"])
                G.add_node(f"head_{name}", type="head", label=name)
        
        # Add edges (simplified connections)
        # Processors to backbone
        for name in processor_names if 'processor_names' in locals() else []:
            G.add_edge(f"proc_{name}", "backbone")
        
        # Backbone to heads
        for name in head_names if 'head_names' in locals() else []:
            G.add_edge("backbone", f"head_{name}")
        
        # Plot the graph
        plt.figure(figsize=(12, 8))
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            if node_type == 'processor':
                node_colors.append('lightblue')
            elif node_type == 'backbone':
                node_colors.append('lightgreen')
            elif node_type == 'head':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightgray')
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=3000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')
        
        # Add labels
        labels = {node: G.nodes[node].get('label', node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Network Topology', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_topology.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Network topology saved to {self.output_dir / 'network_topology.png'}")
    
    def plot_parameter_distribution(self, param_stats: Dict[str, Any]) -> None:
        """Plot parameter distribution statistics."""
        if not param_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Distribution Analysis', fontsize=16)
        
        # 1. Parameter counts by layer
        ax1 = axes[0, 0]
        layer_names = list(param_stats.keys())[:15]  # Top 15 layers
        param_counts = [param_stats[name]["total_params"] for name in layer_names]
        
        ax1.barh(range(len(layer_names)), param_counts)
        ax1.set_yticks(range(len(layer_names)))
        ax1.set_yticklabels([name.split('.')[-1] for name in layer_names])
        ax1.set_xlabel('Parameter Count')
        ax1.set_title('Parameter Count by Layer')
        
        # 2. Parameter value distributions
        ax2 = axes[0, 1]
        all_means = [stats["mean"] for stats in param_stats.values()]
        all_stds = [stats["std"] for stats in param_stats.values()]
        
        ax2.scatter(all_means, all_stds, alpha=0.6)
        ax2.set_xlabel('Parameter Mean')
        ax2.set_ylabel('Parameter Std')
        ax2.set_title('Parameter Statistics Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Zero fraction analysis
        ax3 = axes[1, 0]
        zero_fractions = [stats["zero_fraction"] for stats in param_stats.values()]
        ax3.hist(zero_fractions, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Fraction of Zero Parameters')
        ax3.set_ylabel('Number of Layers')
        ax3.set_title('Sparsity Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Parameter range analysis
        ax4 = axes[1, 1]
        param_ranges = [stats["max"] - stats["min"] for stats in param_stats.values()]
        layer_names_short = [name.split('.')[-1] for name in param_stats.keys()][:10]
        
        ax4.bar(range(len(param_ranges[:10])), param_ranges[:10])
        ax4.set_xticks(range(len(layer_names_short)))
        ax4.set_xticklabels(layer_names_short, rotation=45, ha='right')
        ax4.set_ylabel('Parameter Range')
        ax4.set_title('Parameter Range by Layer')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'parameter_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Parameter distribution saved to {self.output_dir / 'parameter_distribution.png'}")
    
    def plot_layer_dimensions(self, dimensions: Dict[str, Any]) -> None:
        """Plot layer input/output dimensions."""
        layer_outputs = dimensions.get("layer_outputs", {})
        if not layer_outputs:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Layer Dimension Analysis', fontsize=16)
        
        # 1. Output statistics
        ax1 = axes[0, 0]
        layer_names = []
        output_means = []
        output_stds = []
        
        for name, info in layer_outputs.items():
            if isinstance(info, dict) and "output_mean" in info:
                layer_names.append(name.split('.')[-1])
                output_means.append(info["output_mean"])
                output_stds.append(info["output_std"])
        
        if output_means:
            x_pos = range(len(layer_names))
            ax1.bar(x_pos, output_means, alpha=0.7, label='Mean')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(layer_names, rotation=45, ha='right')
            ax1.set_ylabel('Output Mean')
            ax1.set_title('Layer Output Means')
            ax1.grid(True, alpha=0.3)
        
        # 2. Output standard deviations
        ax2 = axes[0, 1]
        if output_stds:
            ax2.bar(x_pos, output_stds, alpha=0.7, color='orange', label='Std')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(layer_names, rotation=45, ha='right')
            ax2.set_ylabel('Output Std')
            ax2.set_title('Layer Output Standard Deviations')
            ax2.grid(True, alpha=0.3)
        
        # 3. Output ranges
        ax3 = axes[1, 0]
        output_ranges = []
        for name, info in layer_outputs.items():
            if isinstance(info, dict) and "output_min" in info and "output_max" in info:
                output_ranges.append(info["output_max"] - info["output_min"])
        
        if output_ranges:
            ax3.bar(x_pos, output_ranges, alpha=0.7, color='green')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(layer_names, rotation=45, ha='right')
            ax3.set_ylabel('Output Range')
            ax3.set_title('Layer Output Ranges')
            ax3.grid(True, alpha=0.3)
        
        # 4. Output shape information
        ax4 = axes[1, 1]
        shape_info = []
        for name, info in layer_outputs.items():
            if isinstance(info, dict) and "output_shape" in info:
                shape = info["output_shape"]
                if len(shape) > 1:
                    total_elements = np.prod(shape[1:])  # Exclude batch dimension
                    shape_info.append((name.split('.')[-1], total_elements))
        
        if shape_info:
            names, elements = zip(*shape_info[:10])  # Top 10
            ax4.barh(range(len(names)), elements)
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels(names)
            ax4.set_xlabel('Output Elements (excluding batch)')
            ax4.set_title('Layer Output Sizes')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            # Fallback: adjust subplot parameters manually
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            # Final fallback: use default spacing
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'layer_dimensions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Layer dimensions saved to {self.output_dir / 'layer_dimensions.png'}")

    def analyze_conv1d_operations(self, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze Conv1D operations in the price processor."""
        results = {}
        
        # Find Conv1D layers in the network
        conv1d_layers = self._find_conv1d_layers()
        
        if not conv1d_layers:
            self.logger.warning("No Conv1D layers found in the network")
            return results
        
        # Analyze each Conv1D layer
        for layer_name, layer in conv1d_layers.items():
            layer_analysis = self._analyze_single_conv1d_layer(layer_name, layer)
            results[layer_name] = layer_analysis
        
        # Analyze filter responses to sample data
        results["filter_responses"] = self._analyze_conv1d_filter_responses(conv1d_layers)
        
        if save_plots:
            self.plot_conv1d_analysis(results)
        
        return results
    
    def _find_conv1d_layers(self) -> Dict[str, nn.Conv1d]:
        """Find all Conv1D layers in the network."""
        conv1d_layers = {}
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv1d):
                conv1d_layers[name] = module
        return conv1d_layers
    
    def _analyze_single_conv1d_layer(self, layer_name: str, layer: nn.Conv1d) -> Dict[str, Any]:
        """Analyze a single Conv1D layer."""
        analysis = {
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "kernel_size": layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size,
            "stride": layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride,
            "padding": layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
            "dilation": layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation,
            "groups": layer.groups,
            "bias": layer.bias is not None
        }
        
        # Analyze filter weights
        weights = layer.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, kernel_size)
        analysis["weight_statistics"] = {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "sparsity": float(np.mean(weights == 0))
        }
        
        # Analyze individual filters
        analysis["filters"] = []
        for i in range(layer.out_channels):
            filter_weights = weights[i]  # Shape: (in_channels, kernel_size)
            filter_analysis = {
                "filter_id": i,
                "mean": float(np.mean(filter_weights)),
                "std": float(np.std(filter_weights)),
                "energy": float(np.sum(filter_weights ** 2)),
                "pattern_type": self._classify_conv1d_pattern(filter_weights)
            }
            analysis["filters"].append(filter_analysis)
        
        return analysis
    
    def _classify_conv1d_pattern(self, filter_weights: np.ndarray) -> str:
        """Classify what type of pattern a Conv1D filter is detecting."""
        # Simple heuristics to classify filter patterns
        kernel_size = filter_weights.shape[-1]
        
        if kernel_size == 1:
            return "pointwise"
        
        # Check for trend detection patterns
        avg_across_channels = np.mean(filter_weights, axis=0)
        
        # Check if it's a trend detector (monotonic)
        if np.all(np.diff(avg_across_channels) > 0):
            return "uptrend_detector"
        elif np.all(np.diff(avg_across_channels) < 0):
            return "downtrend_detector"
        
        # Check for edge detection (high-low pattern)
        if kernel_size >= 3:
            center_idx = kernel_size // 2
            if avg_across_channels[center_idx] > np.mean([avg_across_channels[0], avg_across_channels[-1]]):
                return "peak_detector"
            elif avg_across_channels[center_idx] < np.mean([avg_across_channels[0], avg_across_channels[-1]]):
                return "valley_detector"
        
        # Check for oscillation detection
        if np.std(avg_across_channels) > np.abs(np.mean(avg_across_channels)):
            return "oscillation_detector"
        
        return "general_feature"
    
    def _analyze_conv1d_filter_responses(self, conv1d_layers: Dict[str, nn.Conv1d]) -> Dict[str, Any]:
        """Analyze how Conv1D filters respond to sample price data."""
        responses = {}
        
        # Get sample price data
        sample_obs = self.sample_obs
        
        # Hook to capture activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks on Conv1D layers
        for name, layer in conv1d_layers.items():
            hook = layer.register_forward_hook(hook_fn(name))
            hooks.append(hook)
        
        # Forward pass to capture activations
        with torch.no_grad():
            _ = self.network(sample_obs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activations
        for layer_name, activation in activations.items():
            # activation shape: (batch_size, out_channels, sequence_length)
            responses[layer_name] = {
                "activation_shape": activation.shape,
                "mean_activation": float(np.mean(activation)),
                "std_activation": float(np.std(activation)),
                "max_activation": float(np.max(activation)),
                "sparsity": float(np.mean(activation == 0)),
                "channel_responses": []
            }
            
            # Analyze each output channel
            for channel in range(activation.shape[1]):
                channel_activation = activation[0, channel, :]  # First batch item
                responses[layer_name]["channel_responses"].append({
                    "channel_id": channel,
                    "mean": float(np.mean(channel_activation)),
                    "std": float(np.std(channel_activation)),
                    "max_response_position": int(np.argmax(np.abs(channel_activation))),
                    "response_strength": float(np.max(np.abs(channel_activation)))
                })
        
        return responses
    
    def plot_conv1d_analysis(self, results: Dict[str, Any]) -> None:
        """Plot comprehensive Conv1D analysis."""
        # Filter out non-layer results
        layer_results = {k: v for k, v in results.items() if k != "filter_responses"}
        
        if not layer_results:
            return
        
        n_layers = len(layer_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Conv1D Analysis: Price Pattern Detection', fontsize=16)
        
        # 1. Filter pattern distribution across all layers
        ax1 = axes[0, 0]
        pattern_counts = defaultdict(int)
        for layer_name, layer_data in layer_results.items():
            for filter_data in layer_data["filters"]:
                pattern_counts[filter_data["pattern_type"]] += 1
        
        if pattern_counts:
            patterns, counts = zip(*pattern_counts.items())
            ax1.pie(counts, labels=patterns, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Learned Pattern Types Distribution')
        
        # 2. Filter energy distribution
        ax2 = axes[0, 1]
        all_energies = []
        layer_names = []
        for layer_name, layer_data in layer_results.items():
            energies = [f["energy"] for f in layer_data["filters"]]
            all_energies.extend(energies)
            layer_names.extend([layer_name.split('.')[-1]] * len(energies))
        
        if all_energies:
            # Create box plot of energies by layer
            unique_layers = list(set(layer_names))
            energy_by_layer = [[] for _ in unique_layers]
            for energy, layer in zip(all_energies, layer_names):
                layer_idx = unique_layers.index(layer)
                energy_by_layer[layer_idx].append(energy)
            
            ax2.boxplot(energy_by_layer, labels=unique_layers)
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Filter Energy')
            ax2.set_title('Filter Energy Distribution by Layer')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Kernel size vs pattern type
        ax3 = axes[1, 0]
        kernel_sizes = []
        pattern_types = []
        for layer_name, layer_data in layer_results.items():
            kernel_size = layer_data["kernel_size"]
            for filter_data in layer_data["filters"]:
                kernel_sizes.append(kernel_size)
                pattern_types.append(filter_data["pattern_type"])
        
        if kernel_sizes and pattern_types:
            # Create scatter plot
            unique_patterns = list(set(pattern_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_patterns)))
            
            for i, pattern in enumerate(unique_patterns):
                pattern_kernel_sizes = [ks for ks, pt in zip(kernel_sizes, pattern_types) if pt == pattern]
                pattern_y = [i] * len(pattern_kernel_sizes)
                ax3.scatter(pattern_kernel_sizes, pattern_y, c=[colors[i]], label=pattern, alpha=0.7)
            
            ax3.set_xlabel('Kernel Size')
            ax3.set_ylabel('Pattern Type')
            ax3.set_yticks(range(len(unique_patterns)))
            ax3.set_yticklabels(unique_patterns)
            ax3.set_title('Kernel Size vs Pattern Type')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Activation statistics
        ax4 = axes[1, 1]
        if "filter_responses" in results:
            layer_names = []
            mean_activations = []
            sparsities = []
            
            for layer_name, response_data in results["filter_responses"].items():
                layer_names.append(layer_name.split('.')[-1])
                mean_activations.append(response_data["mean_activation"])
                sparsities.append(response_data["sparsity"])
            
            if layer_names:
                x_pos = np.arange(len(layer_names))
                width = 0.35
                
                ax4_twin = ax4.twinx()
                bars1 = ax4.bar(x_pos - width/2, mean_activations, width, label='Mean Activation', alpha=0.7)
                bars2 = ax4_twin.bar(x_pos + width/2, sparsities, width, label='Sparsity', alpha=0.7, color='orange')
                
                ax4.set_xlabel('Layer')
                ax4.set_ylabel('Mean Activation', color='blue')
                ax4_twin.set_ylabel('Sparsity', color='orange')
                ax4.set_title('Activation Statistics by Layer')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(layer_names, rotation=45, ha='right')
                
                # Combine legends
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_twin.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'conv1d_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Conv1D analysis saved to {self.output_dir / 'conv1d_analysis.png'}")
        
        # Create detailed filter visualization
        self._plot_conv1d_filters_detailed(layer_results)

    def _plot_conv1d_filters_detailed(self, layer_results: Dict[str, Any]) -> None:
        """Create detailed visualization of Conv1D filters."""
        for layer_name, layer_data in layer_results.items():
            if not layer_data["filters"]:
                continue
                
            n_filters = len(layer_data["filters"])
            if n_filters == 0:
                continue
                
            # Create a grid to show individual filters
            cols = min(8, n_filters)
            rows = (n_filters + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 2*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Conv1D Filters: {layer_name}', fontsize=14)
            
            # Get the actual layer to access weights
            layer = None
            for name, module in self.network.named_modules():
                if name == layer_name and isinstance(module, nn.Conv1d):
                    layer = module
                    break
            
            if layer is not None:
                weights = layer.weight.data.cpu().numpy()
                
                for i in range(n_filters):
                    row = i // cols
                    col = i % cols
                    ax = axes[row, col]
                    
                    # Plot filter weights
                    filter_weights = weights[i]  # Shape: (in_channels, kernel_size)
                    
                    if filter_weights.shape[0] == 1:
                        # Single input channel
                        ax.plot(filter_weights[0], 'b-', linewidth=2)
                        ax.set_title(f'Filter {i}\n{layer_data["filters"][i]["pattern_type"]}', fontsize=8)
                    else:
                        # Multiple input channels - show as heatmap
                        im = ax.imshow(filter_weights, cmap='RdBu_r', aspect='auto')
                        ax.set_title(f'Filter {i}\n{layer_data["filters"][i]["pattern_type"]}', fontsize=8)
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=6)
                
                # Hide unused subplots
                for i in range(n_filters, rows * cols):
                    row = i // cols
                    col = i % cols
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            plt.savefig(self.output_dir / f'conv1d_filters_{safe_layer_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Detailed Conv1D filters saved for layer: {layer_name}")

    def analyze_conv2d_operations(self, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze Conv2D operations in the OHLCV processor."""
        results = {}
        
        # Find Conv2D layers in the network
        conv2d_layers = self._find_conv2d_layers()
        
        if not conv2d_layers:
            self.logger.warning("No Conv2D layers found in the network")
            return results
        
        # Analyze each Conv2D layer
        for layer_name, layer in conv2d_layers.items():
            layer_analysis = self._analyze_single_conv2d_layer(layer_name, layer)
            results[layer_name] = layer_analysis
        
        # Analyze filter responses to sample OHLCV data
        results["filter_responses"] = self._analyze_conv2d_filter_responses(conv2d_layers)
        
        if save_plots:
            self.plot_conv2d_analysis(results)
        
        return results
    
    def _find_conv2d_layers(self) -> Dict[str, nn.Conv2d]:
        """Find all Conv2D layers in the network."""
        conv2d_layers = {}
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv2d):
                conv2d_layers[name] = module
        return conv2d_layers
    
    def _analyze_single_conv2d_layer(self, layer_name: str, layer: nn.Conv2d) -> Dict[str, Any]:
        """Analyze a single Conv2D layer."""
        analysis = {
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "groups": layer.groups,
            "bias": layer.bias is not None
        }
        
        # Analyze filter weights
        weights = layer.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, kernel_h, kernel_w)
        analysis["weight_statistics"] = {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "sparsity": float(np.mean(weights == 0))
        }
        
        # Analyze individual filters
        analysis["filters"] = []
        for i in range(layer.out_channels):
            filter_weights = weights[i]  # Shape: (in_channels, kernel_h, kernel_w)
            filter_analysis = {
                "filter_id": i,
                "mean": float(np.mean(filter_weights)),
                "std": float(np.std(filter_weights)),
                "energy": float(np.sum(filter_weights ** 2)),
                "pattern_type": self._classify_conv2d_pattern(filter_weights)
            }
            analysis["filters"].append(filter_analysis)
        
        return analysis
    
    def _classify_conv2d_pattern(self, filter_weights: np.ndarray) -> str:
        """Classify what type of pattern a Conv2D filter is detecting."""
        # filter_weights shape: (in_channels, kernel_h, kernel_w)
        kernel_h, kernel_w = filter_weights.shape[-2:]
        
        if kernel_h == 1 and kernel_w == 1:
            return "pointwise"
        
        # Average across input channels for pattern analysis
        avg_filter = np.mean(filter_weights, axis=0)
        
        # Check for edge detection patterns
        if kernel_h >= 3 and kernel_w >= 3:
            # Check for horizontal edge detection
            top_row = np.mean(avg_filter[0, :])
            bottom_row = np.mean(avg_filter[-1, :])
            if abs(top_row - bottom_row) > 0.5 * np.std(avg_filter):
                return "horizontal_edge_detector"
            
            # Check for vertical edge detection
            left_col = np.mean(avg_filter[:, 0])
            right_col = np.mean(avg_filter[:, -1])
            if abs(left_col - right_col) > 0.5 * np.std(avg_filter):
                return "vertical_edge_detector"
            
            # Check for corner detection
            corners = [avg_filter[0, 0], avg_filter[0, -1], avg_filter[-1, 0], avg_filter[-1, -1]]
            center = avg_filter[kernel_h//2, kernel_w//2]
            if abs(center - np.mean(corners)) > np.std(avg_filter):
                return "corner_detector"
        
        # Check for OHLCV-specific patterns
        if filter_weights.shape[0] >= 5:  # At least 5 input channels (OHLCV + volume)
            # Analyze channel-wise patterns
            channel_means = [np.mean(filter_weights[c]) for c in range(min(5, filter_weights.shape[0]))]
            
            # Check if it focuses on volume (typically last channel)
            if len(channel_means) >= 5 and channel_means[4] > 1.5 * np.mean(channel_means[:4]):
                return "volume_pattern_detector"
            
            # Check if it focuses on price channels (OHLC)
            if np.std(channel_means[:4]) < 0.1 * np.mean(np.abs(channel_means[:4])):
                return "price_pattern_detector"
            
            # Check for candlestick pattern detection
            if kernel_h >= 3:  # Need height for candlestick patterns
                return "candlestick_pattern_detector"
        
        # Check for temporal patterns
        if kernel_w > kernel_h:
            return "temporal_pattern_detector"
        elif kernel_h > kernel_w:
            return "feature_pattern_detector"
        
        return "general_2d_feature"
    
    def _analyze_conv2d_filter_responses(self, conv2d_layers: Dict[str, nn.Conv2d]) -> Dict[str, Any]:
        """Analyze how Conv2D filters respond to sample OHLCV data."""
        responses = {}
        
        # Get sample OHLCV data
        sample_obs = self.sample_obs
        
        # Hook to capture activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks on Conv2D layers
        for name, layer in conv2d_layers.items():
            hook = layer.register_forward_hook(hook_fn(name))
            hooks.append(hook)
        
        # Forward pass to capture activations
        with torch.no_grad():
            _ = self.network(sample_obs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activations
        for layer_name, activation in activations.items():
            # activation shape: (batch_size, out_channels, height, width)
            responses[layer_name] = {
                "activation_shape": activation.shape,
                "mean_activation": float(np.mean(activation)),
                "std_activation": float(np.std(activation)),
                "max_activation": float(np.max(activation)),
                "sparsity": float(np.mean(activation == 0)),
                "channel_responses": []
            }
            
            # Analyze each output channel
            for channel in range(activation.shape[1]):
                channel_activation = activation[0, channel, :, :]  # First batch item
                responses[layer_name]["channel_responses"].append({
                    "channel_id": channel,
                    "mean": float(np.mean(channel_activation)),
                    "std": float(np.std(channel_activation)),
                    "max_response_position": np.unravel_index(np.argmax(np.abs(channel_activation)), channel_activation.shape),
                    "response_strength": float(np.max(np.abs(channel_activation)))
                })
        
        return responses
    
    def plot_conv2d_analysis(self, results: Dict[str, Any]) -> None:
        """Plot comprehensive Conv2D analysis."""
        # Filter out non-layer results
        layer_results = {k: v for k, v in results.items() if k != "filter_responses"}
        
        if not layer_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Conv2D Analysis: OHLCV Pattern Detection', fontsize=16)
        
        # 1. Filter pattern distribution
        ax1 = axes[0, 0]
        pattern_counts = defaultdict(int)
        for layer_name, layer_data in layer_results.items():
            for filter_data in layer_data["filters"]:
                pattern_counts[filter_data["pattern_type"]] += 1
        
        if pattern_counts:
            patterns, counts = zip(*pattern_counts.items())
            ax1.pie(counts, labels=patterns, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Learned 2D Pattern Types')
        
        # 2. Kernel size distribution
        ax2 = axes[0, 1]
        kernel_sizes_h = []
        kernel_sizes_w = []
        for layer_name, layer_data in layer_results.items():
            kernel_h, kernel_w = layer_data["kernel_size"]
            kernel_sizes_h.append(kernel_h)
            kernel_sizes_w.append(kernel_w)
        
        if kernel_sizes_h and kernel_sizes_w:
            ax2.scatter(kernel_sizes_w, kernel_sizes_h, alpha=0.7, s=100)
            ax2.set_xlabel('Kernel Width')
            ax2.set_ylabel('Kernel Height')
            ax2.set_title('Kernel Size Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add annotations for each point
            for i, (w, h) in enumerate(zip(kernel_sizes_w, kernel_sizes_h)):
                ax2.annotate(f'L{i}', (w, h), xytext=(5, 5), textcoords='offset points')
        
        # 3. Filter energy vs pattern type
        ax3 = axes[1, 0]
        pattern_energies = defaultdict(list)
        for layer_name, layer_data in layer_results.items():
            for filter_data in layer_data["filters"]:
                pattern_energies[filter_data["pattern_type"]].append(filter_data["energy"])
        
        if pattern_energies:
            patterns = list(pattern_energies.keys())
            energies = [pattern_energies[pattern] for pattern in patterns]
            ax3.boxplot(energies, labels=patterns)
            ax3.set_xlabel('Pattern Type')
            ax3.set_ylabel('Filter Energy')
            ax3.set_title('Energy Distribution by Pattern Type')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Activation statistics
        ax4 = axes[1, 1]
        if "filter_responses" in results:
            layer_names = []
            mean_activations = []
            sparsities = []
            
            for layer_name, response_data in results["filter_responses"].items():
                layer_names.append(layer_name.split('.')[-1])
                mean_activations.append(response_data["mean_activation"])
                sparsities.append(response_data["sparsity"])
            
            if layer_names:
                x_pos = np.arange(len(layer_names))
                width = 0.35
                
                ax4_twin = ax4.twinx()
                ax4.bar(x_pos - width/2, mean_activations, width, label='Mean Activation', alpha=0.7)
                ax4_twin.bar(x_pos + width/2, sparsities, width, label='Sparsity', alpha=0.7, color='orange')
                
                ax4.set_xlabel('Layer')
                ax4.set_ylabel('Mean Activation', color='blue')
                ax4_twin.set_ylabel('Sparsity', color='orange')
                ax4.set_title('Activation Statistics by Layer')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(layer_names, rotation=45, ha='right')
                
                # Combine legends
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_twin.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Use more robust layout handling
        try:
            plt.tight_layout()
        except UserWarning:
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        except Exception:
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        plt.savefig(self.output_dir / 'conv2d_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Conv2D analysis saved to {self.output_dir / 'conv2d_analysis.png'}")
        
        # Create detailed filter visualization
        self._plot_conv2d_filters_detailed(layer_results)

    def _plot_conv2d_filters_detailed(self, layer_results: Dict[str, Any]) -> None:
        """Create detailed visualization of Conv2D filters."""
        for layer_name, layer_data in layer_results.items():
            if not layer_data["filters"]:
                continue
                
            n_filters = len(layer_data["filters"])
            if n_filters == 0:
                continue
                
            # Create a grid to show individual filters
            cols = min(8, n_filters)
            rows = (n_filters + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 2*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Conv2D Filters: {layer_name}', fontsize=14)
            
            # Get the actual layer to access weights
            layer = None
            for name, module in self.network.named_modules():
                if name == layer_name and isinstance(module, nn.Conv2d):
                    layer = module
                    break
            
            if layer is not None:
                weights = layer.weight.data.cpu().numpy()
                
                for i in range(n_filters):
                    row = i // cols
                    col = i % cols
                    ax = axes[row, col]
                    
                    # Plot filter weights
                    filter_weights = weights[i]  # Shape: (in_channels, kernel_h, kernel_w)
                    
                    if filter_weights.shape[0] == 1:
                        # Single input channel
                        ax.plot(filter_weights[0], 'b-', linewidth=2)
                        ax.set_title(f'Filter {i}\n{layer_data["filters"][i]["pattern_type"]}', fontsize=8)
                    else:
                        # Multiple input channels - show as heatmap
                        im = ax.imshow(filter_weights, cmap='RdBu_r', aspect='auto')
                        ax.set_title(f'Filter {i}\n{layer_data["filters"][i]["pattern_type"]}', fontsize=8)
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=6)
                
                # Hide unused subplots
                for i in range(n_filters, rows * cols):
                    row = i // cols
                    col = i % cols
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            plt.savefig(self.output_dir / f'conv2d_filters_{safe_layer_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Detailed Conv2D filters saved for layer: {layer_name}")

    def analyze_onnx_export(self, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze and export network to ONNX format for advanced topology visualization."""
        results = {}
        
        try:
            # Export network to ONNX
            onnx_path = self.output_dir / f"{self.agent.get_model_name()}_network.onnx"
            export_success = self._export_to_onnx(onnx_path)
            
            if export_success:
                results["onnx_export"] = {
                    "success": True,
                    "onnx_path": str(onnx_path),
                    "file_size": onnx_path.stat().st_size if onnx_path.exists() else 0
                }
                
                # Analyze ONNX model structure
                results["onnx_analysis"] = self._analyze_onnx_model(onnx_path)
                
                if save_plots:
                    self._create_onnx_visualization_instructions(onnx_path)
            else:
                results["onnx_export"] = {
                    "success": False,
                    "error": "Failed to export network to ONNX format"
                }
                
        except Exception as e:
            results["onnx_export"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.warning(f"ONNX export failed: {e}")
        
        return results
    
    def _export_to_onnx(self, onnx_path: Path) -> bool:
        """Export the network to ONNX format."""
        try:
            # Prepare sample input
            self.logger.info("Preparing sample input for ONNX export...")
            sample_input = self._prepare_onnx_input()
            self.logger.info("Sample input prepared successfully")
            
            # Create ONNX-compatible wrapper that accepts multiple tensor inputs
            self.logger.info("Creating ONNX-compatible wrapper...")
            onnx_wrapper, input_tuple, input_names = self._create_onnx_wrapper(sample_input)
            
            # Set wrapper to evaluation mode
            onnx_wrapper.eval()
            self.logger.info("ONNX wrapper set to evaluation mode")
            
            # Test forward pass first to catch any issues
            self.logger.info("Testing forward pass with ONNX wrapper...")
            with torch.no_grad():
                try:
                    test_output = onnx_wrapper(*input_tuple)
                    self.logger.info(f"Forward pass successful, output type: {type(test_output)}")
                    if isinstance(test_output, torch.Tensor):
                        self.logger.info(f"Output shape: {test_output.shape}, dtype: {test_output.dtype}")
                    elif isinstance(test_output, dict):
                        self.logger.info(f"Output keys: {list(test_output.keys())}")
                        for key, value in test_output.items():
                            if isinstance(value, torch.Tensor):
                                self.logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                except Exception as e:
                    self.logger.error(f"Forward pass failed: {e}")
                    return False
            
            # Export to ONNX
            self.logger.info("Starting ONNX export...")
            torch.onnx.export(
                onnx_wrapper,
                input_tuple,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['action_probs'],
                dynamic_axes={
                    # Allow dynamic batch size for all inputs
                    **{name: {0: 'batch_size'} for name in input_names},
                    'action_probs': {0: 'batch_size'}
                },
                verbose=False  # Reduce verbosity to avoid clutter
            )
            
            self.logger.info(f"Successfully exported network to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to export to ONNX: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _create_onnx_wrapper(self, sample_input):
        """Create an ONNX-compatible wrapper that accepts multiple tensor inputs instead of a dictionary."""
        
        # Sort keys for consistent ordering
        sorted_keys = sorted(sample_input.keys())
        
        class ONNXWrapper(nn.Module):
            def __init__(self, original_network, input_keys):
                super().__init__()
                self.network = original_network
                self.input_keys = input_keys
                
            def forward(self, *args):
                # Reconstruct dictionary from positional arguments
                observations = {}
                for i, key in enumerate(self.input_keys):
                    observations[key] = args[i]
                
                # Forward through original network
                output = self.network(observations)
                
                # Return the main output (action_probs for this network)
                if isinstance(output, dict):
                    if 'action_probs' in output:
                        return output['action_probs']
                    elif 'value' in output:
                        return output['value']
                    else:
                        # Return the first available output
                        return list(output.values())[0]
                else:
                    return output
        
        # Create wrapper
        wrapper = ONNXWrapper(self.network, sorted_keys)
        
        # Create input tuple in the same order as sorted_keys
        input_tuple = tuple(sample_input[key] for key in sorted_keys)
        
        # Create input names list
        input_names = sorted_keys
        
        self.logger.info(f"Created ONNX wrapper with inputs: {input_names}")
        self.logger.info(f"Input shapes: {[tensor.shape for tensor in input_tuple]}")
        
        return wrapper, input_tuple, input_names
    
    def _prepare_onnx_input(self):
        """Prepare sample input for ONNX export."""
        # Use the sample observation but ensure it's properly formatted
        sample_input = {}
        
        # Ensure we have at least one observation
        # Check if sample_obs is a dictionary and has keys
        if not isinstance(self.sample_obs, dict) or not self.sample_obs:
            raise ValueError("No sample observations available for ONNX export")
        
        self.logger.info(f"Processing {len(self.sample_obs)} observation keys for ONNX export")
        
        for key, value in self.sample_obs.items():
            self.logger.info(f"Processing observation key '{key}': type={type(value)}")
            
            if isinstance(value, torch.Tensor):
                self.logger.info(f"  Tensor details: shape={value.shape}, dim={value.dim()}, dtype={value.dtype}, device={value.device}")
                
                # Ensure tensor is on the same device as the network and has proper shape
                tensor = value.to(self.device)  # Move to network device
                if tensor.dim() == 0:  # Scalar tensor
                    self.logger.info(f"  Converting 0-d tensor to 2-d: {tensor} -> shape will be (1, 1)")
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                elif tensor.dim() == 1:  # 1D tensor
                    self.logger.info(f"  Converting 1-d tensor to 2-d: shape {tensor.shape} -> {(1,) + tensor.shape}")
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() >= 2:
                    # Check if we already have a batch dimension
                    if tensor.shape[0] == 1:
                        # Already has batch dimension
                        self.logger.info(f"  Tensor already has batch dimension: {tensor.shape}")
                        sample_input[key] = tensor
                        continue
                    else:
                        # Add batch dimension
                        self.logger.info(f"  Adding batch dimension: shape {tensor.shape} -> {(1,) + tensor.shape}")
                        tensor = tensor.unsqueeze(0)
                
                sample_input[key] = tensor
                self.logger.info(f"  Final tensor shape for '{key}': {tensor.shape}, device: {tensor.device}")
                
            else:
                self.logger.info(f"  Converting non-tensor value: {value}")
                
                # Convert to tensor if not already
                if isinstance(value, (int, float)):
                    tensor = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
                    self.logger.info(f"  Converted scalar to tensor: shape {tensor.shape}")
                elif isinstance(value, np.ndarray):
                    self.logger.info(f"  Converting numpy array: shape={value.shape}, dtype={value.dtype}")
                    tensor = torch.FloatTensor(value).to(self.device)
                    if tensor.dim() == 0:  # Scalar
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                        self.logger.info(f"  Converted 0-d numpy to tensor: shape {tensor.shape}")
                    elif tensor.dim() == 1:  # 1D
                        tensor = tensor.unsqueeze(0)
                        self.logger.info(f"  Converted 1-d numpy to tensor: shape {tensor.shape}")
                    elif tensor.dim() >= 2 and tensor.shape[0] != 1:
                        tensor = tensor.unsqueeze(0)
                        self.logger.info(f"  Added batch dim to numpy tensor: shape {tensor.shape}")
                else:
                    # Try to convert to float and create tensor
                    try:
                        tensor = torch.FloatTensor([float(value)]).unsqueeze(0).to(self.device)
                        self.logger.info(f"  Converted other type to tensor: shape {tensor.shape}")
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert observation key '{key}' with value {value} to tensor")
                        continue
                
                sample_input[key] = tensor
                self.logger.info(f"  Final tensor shape for '{key}': {tensor.shape}, device: {tensor.device}")
        
        # Verify we have valid input
        if not sample_input:
            raise ValueError("Failed to create valid sample input for ONNX export")
        
        # Log the prepared input shapes for debugging
        self.logger.info("Final prepared ONNX input shapes:")
        for key, tensor in sample_input.items():
            self.logger.info(f"  {key}: {tensor.shape}, device: {tensor.device}")
        
        return sample_input
    
    def _analyze_onnx_model(self, onnx_path: Path) -> Dict[str, Any]:
        """Analyze the exported ONNX model structure."""
        analysis = {}
        
        try:
            import onnx
            
            # Load ONNX model
            model = onnx.load(str(onnx_path))
            
            # Basic model info
            analysis["model_info"] = {
                "ir_version": model.ir_version,
                "producer_name": model.producer_name,
                "producer_version": model.producer_version,
                "domain": model.domain,
                "model_version": model.model_version,
                "doc_string": model.doc_string
            }
            
            # Graph analysis
            graph = model.graph
            analysis["graph_info"] = {
                "name": graph.name,
                "num_nodes": len(graph.node),
                "num_inputs": len(graph.input),
                "num_outputs": len(graph.output),
                "num_initializers": len(graph.initializer),
                "num_value_infos": len(graph.value_info)
            }
            
            # Node type distribution
            node_types = {}
            for node in graph.node:
                node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
            analysis["node_types"] = node_types
            
            # Input/Output information
            analysis["inputs"] = []
            for input_tensor in graph.input:
                input_info = {
                    "name": input_tensor.name,
                    "type": input_tensor.type.tensor_type.elem_type,
                    "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                }
                analysis["inputs"].append(input_info)
            
            analysis["outputs"] = []
            for output_tensor in graph.output:
                output_info = {
                    "name": output_tensor.name,
                    "type": output_tensor.type.tensor_type.elem_type,
                    "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                }
                analysis["outputs"].append(output_info)
            
            self.logger.info("Successfully analyzed ONNX model structure")
            
        except ImportError:
            analysis["error"] = "ONNX package not available. Install with: pip install onnx"
            self.logger.warning("ONNX package not available for model analysis")
        except Exception as e:
            analysis["error"] = str(e)
            self.logger.error(f"Failed to analyze ONNX model: {e}")
        
        return analysis
    
    def _create_onnx_visualization_instructions(self, onnx_path: Path) -> None:
        """Create instructions for visualizing the exported ONNX model."""
        # This method should be implemented to create and save instructions for visualizing the ONNX model
        # using tools like Netron or other visualization tools
        self.logger.info(f"Instructions for visualizing the exported ONNX model: {onnx_path}")
        # Placeholder for actual implementation
    