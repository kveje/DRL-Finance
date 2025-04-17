import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import shutil
from pathlib import Path

class ModelManager:
    """
    Manages model saving, backtesting, and visualization storage.
    Handles automatic model checkpoints, backtesting on validation data,
    and storage of training visualizations.
    """
    
    def __init__(
        self,
        base_dir: str = "models/saved",
        save_interval: int = 1000,  # Save every N steps
        backtest_interval: int = 5000,  # Backtest every N steps
        max_saves: int = 10,  # Maximum number of checkpoints to keep
        save_visualizations: bool = True
    ):
        """
        Initialize the model manager.
        
        Args:
            base_dir: Base directory for saving models and results
            save_interval: Number of steps between model saves
            backtest_interval: Number of steps between backtests
            max_saves: Maximum number of checkpoints to keep
            save_visualizations: Whether to save training visualizations
        """
        self.base_dir = Path(base_dir)
        self.save_interval = save_interval
        self.backtest_interval = backtest_interval
        self.max_saves = max_saves
        self.save_visualizations = save_visualizations
        
        # Create base directory structure
        self._create_directory_structure()
        
        # Initialize tracking
        self.last_save_step = 0
        self.last_backtest_step = 0
        self.checkpoints = []
    
    def _create_directory_structure(self):
        """Create the directory structure for saving models and results."""
        # Create base directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
        (self.base_dir / "backtests").mkdir(exist_ok=True)
        (self.base_dir / "visualizations").mkdir(exist_ok=True)
        (self.base_dir / "metrics").mkdir(exist_ok=True)
        
        # Create metadata file if it doesn't exist
        metadata_path = self.base_dir / "metadata.json"
        if not metadata_path.exists():
            with open(metadata_path, 'w') as f:
                json.dump({
                    "checkpoints": [],
                    "backtests": [],
                    "metrics": {}
                }, f, indent=4)
    
    def should_save(self, current_step: int) -> bool:
        """Check if we should save the model at the current step."""
        return current_step - self.last_save_step >= self.save_interval
    
    def should_backtest(self, current_step: int) -> bool:
        """Check if we should run a backtest at the current step."""
        return current_step - self.last_backtest_step >= self.backtest_interval
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        current_step: int,
        metrics: Dict[str, Any],
        visualizer: Optional[Any] = None
    ):
        """
        Save a model checkpoint and associated data.
        
        Args:
            model: The model to save
            optimizer: The optimizer to save
            current_step: Current training step
            metrics: Dictionary of training metrics
            visualizer: Optional visualizer to save plots from
        """
        # Create checkpoint directory
        checkpoint_dir = self.base_dir / "checkpoints" / f"step_{current_step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model and optimizer state
        torch.save({
            'step': current_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_dir / "checkpoint.pt")
        
        # Save metrics
        with open(checkpoint_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save visualization if provided
        if self.save_visualizations and visualizer is not None:
            visualizer.save_figure(str(checkpoint_dir / "visualization.png"))
        
        # Update metadata
        self._update_metadata(current_step, checkpoint_dir)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        self.last_save_step = current_step
    
    def _update_metadata(self, current_step: int, checkpoint_dir: Path):
        """Update the metadata file with new checkpoint information."""
        metadata_path = self.base_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        checkpoint_info = {
            "step": current_step,
            "path": str(checkpoint_dir.relative_to(self.base_dir)),
            "timestamp": datetime.now().isoformat()
        }
        
        metadata["checkpoints"].append(checkpoint_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_saves limit."""
        metadata_path = self.base_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        checkpoints = metadata["checkpoints"]
        if len(checkpoints) > self.max_saves:
            # Sort checkpoints by step
            checkpoints.sort(key=lambda x: x["step"])
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.max_saves]:
                checkpoint_path = self.base_dir / checkpoint["path"]
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
            
            # Update metadata
            metadata["checkpoints"] = checkpoints[-self.max_saves:]
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    
    def load_checkpoint(self, step: Optional[int] = None) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            step: Step number to load. If None, loads the latest checkpoint.
            
        Returns:
            Dictionary containing the checkpoint data
        """
        metadata_path = self.base_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if not metadata["checkpoints"]:
            raise ValueError("No checkpoints available")
        
        if step is None:
            # Load latest checkpoint
            checkpoint_info = metadata["checkpoints"][-1]
        else:
            # Find checkpoint with matching step
            checkpoint_info = next(
                (c for c in metadata["checkpoints"] if c["step"] == step),
                None
            )
            if checkpoint_info is None:
                raise ValueError(f"No checkpoint found for step {step}")
        
        checkpoint_path = self.base_dir / checkpoint_info["path"] / "checkpoint.pt"
        return torch.load(checkpoint_path)
    
    def save_backtest_results(
        self,
        step: int,
        results: Dict[str, Any],
        visualizer: Optional[Any] = None
    ):
        """
        Save backtest results and visualizations.
        
        Args:
            step: Current training step
            results: Dictionary of backtest results
            visualizer: Optional visualizer to save plots from
        """
        # Create backtest directory
        backtest_dir = self.base_dir / "backtests" / f"step_{step}"
        backtest_dir.mkdir(exist_ok=True)
        
        # Save results
        with open(backtest_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save visualization if provided
        if self.save_visualizations and visualizer is not None:
            visualizer.save_figure(str(backtest_dir / "backtest_visualization.png"))
        
        # Update metadata
        metadata_path = self.base_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        backtest_info = {
            "step": step,
            "path": str(backtest_dir.relative_to(self.base_dir)),
            "timestamp": datetime.now().isoformat()
        }
        
        metadata["backtests"].append(backtest_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.last_backtest_step = step 