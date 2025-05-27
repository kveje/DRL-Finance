"""
Main Network Visualizer

This module provides the main interface for loading experiments and 
coordinating different types of neural network visualizations.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json

from managers.experiment_manager import ExperimentManager
from models.agents.agent_factory import AgentFactory
from environments.trading_env import TradingEnv
from utils.logger import Logger

from .architecture_analyzer import ArchitectureAnalyzer
from .sensitivity_analyzer import SensitivityAnalyzer
from .diagnostic_plotter import DiagnosticPlotter


class NetworkVisualizer:
    """
    Main interface for neural network visualization and analysis.
    
    This class loads trained experiments and provides access to various
    visualization and analysis tools for understanding the trained networks.
    """
    
    def __init__(self, experiment_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the network visualizer.
        
        Args:
            experiment_dir: Path to the experiment directory
            output_dir: Directory to save visualizations (defaults to experiment_dir/network_analysis)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = self.experiment_dir.name
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = self.experiment_dir / "network_analysis"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        Logger.configure(
            EXPERIMENT_NAME=f"network_viz_{self.experiment_name}",
            EXPERIMENT_DIR=str(self.experiment_dir)
        )
        self.logger = Logger.get_logger("network_visualizer")
        
        # Initialize components
        self.experiment_data = None
        self.agent = None
        self.train_env = None
        self.val_env = None
        
        # Initialize analyzers
        self.architecture_analyzer = None
        self.sensitivity_analyzer = None
        self.diagnostic_plotter = None
        
        self.logger.info(f"Initialized NetworkVisualizer for experiment: {self.experiment_name}")
    
    def load_experiment(self) -> Dict[str, Any]:
        """
        Load experiment data and initialize analyzers.
        
        Returns:
            Dictionary containing experiment information
        """
        self.logger.info(f"Loading experiment from: {self.experiment_dir}")
        
        # Load experiment data using ExperimentManager
        self.experiment_data = ExperimentManager.load_experiment(str(self.experiment_dir))
        
        # Extract configuration and data
        configs = self.experiment_data["config"]
        train_data = self.experiment_data["train_data"]
        val_data = self.experiment_data["val_data"]
        raw_train_data = self.experiment_data["raw_train_data"]
        raw_val_data = self.experiment_data["raw_val_data"]
        data_info = self.experiment_data["data_info"]
        
        # Extract columns config from data info
        columns = data_info.get("columns_mapping", {
            "ticker": "ticker",
            "price": "close",
            "day": "day",
            "ohlcv": ["open", "high", "low", "close", "volume"],
            "tech_cols": [col for col in train_data.columns if col not in ["ticker", "day", "open", "high", "low", "close", "volume", "date", "timestamp"]]
        })
        
        # Create environments
        self.train_env = TradingEnv(
            processed_data=train_data,
            raw_data=raw_train_data,
            columns=columns,
            env_params=configs["environment"].get("env_params", {}),
            friction_params=configs["environment"].get("friction_params", {}),
            constraint_params=configs["environment"].get("constraint_params", {}),
            reward_params=configs["environment"].get("reward_params", {}),
            processor_configs=configs["environment"].get("processor_configs", {}),
            render_mode=None,
        )
        
        self.val_env = TradingEnv(
            processed_data=val_data,
            raw_data=raw_val_data,
            columns=columns,
            env_params=configs["environment"].get("env_params", {}),
            friction_params=configs["environment"].get("friction_params", {}),
            constraint_params=configs["environment"].get("constraint_params", {}),
            reward_params=configs["environment"].get("reward_params", {}),
            processor_configs=configs["environment"].get("processor_configs", {}),
            render_mode=None,
        )
        
        # Create agent
        agent_config = configs["agent"]
        agent_type = configs["experiment"].get("agent_type", "dqn").lower()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent = AgentFactory.create_agent(
            agent_type=agent_type,
            update_frequency=configs["experiment"].get("update_frequency", 1),
            **agent_config
        )
        
        # Load model weights using checkpoint manager directly
        from managers.checkpoint_manager import CheckpointManager
        checkpoint_manager = CheckpointManager(
            str(self.experiment_dir),
            self.agent,  # Pass the agent so it can load weights
            logger=self.logger
        )
        
        # Try to get the latest checkpoint
        latest_model_path = checkpoint_manager.get_latest_checkpoint()
        
        if latest_model_path and os.path.exists(latest_model_path):
            self.logger.info(f"Loading model weights from: {latest_model_path}")
            try:
                # Use weights_only=False for our trusted experiment checkpoints
                # This is safe since we're loading our own saved models
                checkpoint = torch.load(latest_model_path, map_location=device, weights_only=False)
                self.agent.load_state_dict(checkpoint["agent_state"])
                self.logger.info("Successfully loaded model weights")
            except Exception as e:
                self.logger.error(f"Failed to load model weights: {e}")
                self.logger.warning("Using randomly initialized weights")
        else:
            self.logger.warning("No model checkpoint found - using randomly initialized weights")
            if latest_model_path:
                self.logger.warning(f"Checkpoint path exists but file not found: {latest_model_path}")
            else:
                self.logger.warning("No checkpoint path returned from checkpoint manager")
        
        # Set agent to evaluation mode
        self.agent.eval()
        
        # Initialize analyzers
        self.architecture_analyzer = ArchitectureAnalyzer(
            agent=self.agent,
            train_env=self.train_env,
            output_dir=self.output_dir / "architecture",
            logger=self.logger
        )
        
        self.sensitivity_analyzer = SensitivityAnalyzer(
            agent=self.agent,
            train_env=self.train_env,
            val_env=self.val_env,
            output_dir=self.output_dir / "sensitivity",
            logger=self.logger
        )
        
        self.diagnostic_plotter = DiagnosticPlotter(
            agent=self.agent,
            train_env=self.train_env,
            val_env=self.val_env,
            experiment_data=self.experiment_data,
            output_dir=self.output_dir / "diagnostics",
            logger=self.logger
        )
        
        self.logger.info("Successfully loaded experiment and initialized analyzers")
        
        return {
            "experiment_name": self.experiment_name,
            "agent_type": agent_type,
            "network_config": agent_config.get("network_config", {}),
            "n_assets": self.train_env.n_assets,
            "window_size": self.train_env.window_size,
            "train_data_shape": train_data.shape,
            "val_data_shape": val_data.shape,
            "model_loaded": latest_model_path is not None and os.path.exists(latest_model_path),
            "latest_model_path": latest_model_path
        }
    
    def analyze_architecture(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Analyze and visualize the network architecture.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing architecture analysis results
        """
        if self.architecture_analyzer is None:
            raise RuntimeError("Must call load_experiment() first")
        
        self.logger.info("Analyzing network architecture...")
        return self.architecture_analyzer.analyze_full_architecture(save_plots=save_plots)
    
    def analyze_sensitivity(
        self,
        n_samples: int = 1000,
        perturbation_scales: List[float] = [0.01, 0.05, 0.1, 0.2],
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on the network.
        
        Args:
            n_samples: Number of samples to use for analysis
            perturbation_scales: List of perturbation scales to test
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        if self.sensitivity_analyzer is None:
            raise RuntimeError("Must call load_experiment() first")
        
        self.logger.info("Performing sensitivity analysis...")
        return self.sensitivity_analyzer.analyze_full_sensitivity(
            n_samples=n_samples,
            perturbation_scales=perturbation_scales,
            save_plots=save_plots
        )
    
    def create_diagnostic_plots(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Create diagnostic plots for theoretical analysis.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing diagnostic analysis results
        """
        if self.diagnostic_plotter is None:
            raise RuntimeError("Must call load_experiment() first")
        
        self.logger.info("Creating diagnostic plots...")
        return self.diagnostic_plotter.create_all_diagnostics(save_plots=save_plots)
    
    def run_analysis(self, analysis_type: str = "full", **kwargs) -> Dict[str, Any]:
        """
        Run network analysis.
        
        Args:
            analysis_type: Type of analysis to run
                - "full": Complete analysis including all components
                - "structure": Basic network structure analysis
                - "conv1d": Conv1D operations analysis for price processors
                - "conv2d": Conv2D operations analysis for OHLCV processors
                - "onnx": ONNX export and topology visualization
            **kwargs: Additional arguments for specific analysis types
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Running {analysis_type} analysis...")
        
        if analysis_type == "full":
            # Run complete analysis
            results = self.architecture_analyzer.analyze_full_architecture(save_plots=True)
            
            # Always run sensitivity analysis
            sensitivity_results = self.sensitivity_analyzer.analyze_full_sensitivity(
                n_samples=kwargs.get("n_samples", 100)
            )
            results["sensitivity"] = sensitivity_results
            
            # Always run diagnostic analysis
            diagnostic_results = self.diagnostic_plotter.create_all_diagnostics()
            results["diagnostics"] = diagnostic_results
            
        elif analysis_type == "structure":
            # Basic structure analysis
            results = {
                "structure": self.architecture_analyzer.analyze_network_structure(),
                "dimensions": self.architecture_analyzer.analyze_layer_dimensions(),
                "parameters": self.architecture_analyzer.analyze_parameter_distribution(),
                "data_flow": self.architecture_analyzer.analyze_data_flow()
            }
            
            # Create basic plots
            self.architecture_analyzer.plot_architecture_summary(results)
            self.architecture_analyzer.plot_network_topology()
            
        elif analysis_type == "conv1d":
            # Conv1D analysis only
            results = self.architecture_analyzer.analyze_conv1d_operations(save_plots=True)
            
        elif analysis_type == "conv2d":
            # Conv2D analysis only
            results = self.architecture_analyzer.analyze_conv2d_operations(save_plots=True)
            
        elif analysis_type == "onnx":
            # ONNX export and analysis only
            results = self.architecture_analyzer.analyze_onnx_export(save_plots=True)
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        self.logger.info(f"Analysis completed: {analysis_type}")
        return results
    
    def _save_analysis_summary(self, results: Dict[str, Any], analysis_type: str) -> None:
        """Save a summary report of the analysis."""
        summary_file = self.output_dir / f"{analysis_type}_analysis_summary.json"
        
        # Create a serializable version of results
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: str(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = str(value) if isinstance(value, (np.ndarray, torch.Tensor)) else value
        
        with open(summary_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis summary saved to: {summary_file}")
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about the loaded experiment."""
        if self.experiment_data is None:
            return {"status": "No experiment loaded"}
        
        return {
            "experiment_name": self.experiment_name,
            "experiment_dir": str(self.experiment_dir),
            "output_dir": str(self.output_dir),
            "agent_loaded": self.agent is not None,
            "environments_loaded": self.train_env is not None and self.val_env is not None,
            "analyzers_initialized": all([
                self.architecture_analyzer is not None,
                self.sensitivity_analyzer is not None,
                self.diagnostic_plotter is not None
            ])
        } 