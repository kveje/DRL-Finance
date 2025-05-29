"""Unified network framework for trading agents."""
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

from .processors import (
    OHLCVProcessor,
    TechProcessor,
    PriceProcessor,
    PositionProcessor,
    CashProcessor,
    AffordabilityProcessor,
    CurrentPriceProcessor
)
from .backbones.mlp_backbone import MLPBackbone
from .heads import (
    ParametricValueHead,
    BayesianValueHead,
    ParametricDiscreteHead,
    BayesianDiscreteHead,
    ParametricConfidenceHead,
    BayesianConfidenceHead
)

class UnifiedNetwork(nn.Module):
    """Unified network that combines processors, backbone, and heads."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize the unified network.
        
        Args:
            config: Configuration dictionary containing:
                n_assets: Number of assets in the trading environment
                window_size: Size of the observation window for time series data
                processors: Dict of processor configurations
                backbone: Dict of backbone configuration
                heads: Dict of head configurations
            device: Device to run the network on
        """
        super().__init__()
        self.config = config
        self.device = device
        self.n_assets = config.get("n_assets", 3)  # Default to 3 assets if not specified
        self.window_size = config.get("window_size", 20)  # Default to 20 timesteps if not specified
        
        # Initialize components
        self.processors = nn.ModuleDict()
        self._init_processors()
        
        # Calculate total input dimension
        total_input_dim = sum(
            processor.get_output_dim()
            for processor in self.processors.values()
            if processor is not None  # Only include initialized processors
        )
        
        # Initialize backbone
        self.backbone = self._init_backbone(total_input_dim)
        
        # Initialize heads
        self.heads = nn.ModuleDict()
        self._init_heads()
    
    def _init_processors(self):
        """Initialize enabled processors based on configuration."""
        processor_configs = self.config.get("processors", {})
        
        # OHLCV Processor
        if processor_configs.get("ohlcv", {}).get("enabled", False):
            self.processors["ohlcv"] = OHLCVProcessor(
                window_size=self.window_size,
                hidden_dim=processor_configs["ohlcv"].get("hidden_dim", 256),
                device=self.device,
                n_assets=self.n_assets
            )

        # Tech Processor
        if processor_configs.get("tech", {}).get("enabled", False):
            self.processors["tech"] = TechProcessor(
                n_assets=self.n_assets,
                tech_dim=processor_configs["tech"].get("tech_dim", 20),
                hidden_dim=processor_configs["tech"].get("hidden_dim", 64),
                device=self.device
            )

        # Price Processor
        if processor_configs.get("price", {}).get("enabled", False):
            self.processors["price"] = PriceProcessor(
                window_size=self.window_size,
                n_assets=self.n_assets,
                hidden_dim=processor_configs["price"].get("hidden_dim", 128),
                device=self.device
            )

        # Position Processor
        if processor_configs.get("position", {}).get("enabled", False):
            self.processors["position"] = PositionProcessor(
                n_assets=self.n_assets,
                hidden_dim=processor_configs["position"].get("hidden_dim", 64),
                device=self.device
            )

        # Cash Processor
        if processor_configs.get("cash", {}).get("enabled", False):
            self.processors["cash"] = CashProcessor(
                input_dim=processor_configs["cash"].get("input_dim", 2),
                hidden_dim=processor_configs["cash"].get("hidden_dim", 32),
                device=self.device
            )

        # Affordability Processor
        if processor_configs.get("affordability", {}).get("enabled", False):
            self.processors["affordability"] = AffordabilityProcessor(
                n_assets=self.n_assets,
                hidden_dim=processor_configs["affordability"].get("hidden_dim", 64),
                device=self.device
            )

        # Current Price Processor
        if processor_configs.get("current_price", {}).get("enabled", False):
            self.processors["current_price"] = CurrentPriceProcessor(
                n_assets=self.n_assets,
                hidden_dim=processor_configs["current_price"].get("hidden_dim", 64),
                device=self.device
            )
    
    def _init_backbone(self, input_dim: int) -> nn.Module:
        """Initialize the backbone based on configuration."""
        backbone_config = self.config.get("backbone", {})
        backbone_type = backbone_config.get("type", "mlp")
        
        if backbone_type == "mlp":
            return MLPBackbone(
                input_dim=input_dim,
                config=backbone_config,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def _init_heads(self):
        """Initialize enabled heads based on configuration."""
        head_configs = self.config.get("heads", {})
        backbone_output_dim = self.backbone.get_output_dim()
        
        # Value Head
        if head_configs.get("value", {}).get("enabled", False):
            value_config = head_configs["value"]
            head_type = value_config.get("type", "parametric")
            
            if head_type == "parametric":
                self.heads["value"] = ParametricValueHead(
                    input_dim=backbone_output_dim,
                    hidden_dim=value_config.get("hidden_dim", 128),
                    n_assets=self.n_assets,
                    device=self.device
                )
            elif head_type == "bayesian":
                self.heads["value"] = BayesianValueHead(
                    input_dim=backbone_output_dim,
                    hidden_dim=value_config.get("hidden_dim", 128),
                    n_assets=self.n_assets,
                    device=self.device,
                    sampling_strategy=value_config.get("sampling_strategy", "thompson")
                )
            else:
                raise ValueError(f"Unsupported value head type: {head_type}")
        
        # Discrete Head
        if head_configs.get("discrete", {}).get("enabled", False):
            discrete_config = head_configs["discrete"]
            head_type = discrete_config.get("type", "parametric")
            
            if head_type == "parametric":
                self.heads["discrete"] = ParametricDiscreteHead(
                    input_dim=backbone_output_dim,
                    hidden_dim=discrete_config.get("hidden_dim", 128),
                    n_assets=self.n_assets,
                    device=self.device
                )
            elif head_type == "bayesian":
                self.heads["discrete"] = BayesianDiscreteHead(
                    input_dim=backbone_output_dim,
                    hidden_dim=discrete_config.get("hidden_dim", 128),
                    n_assets=self.n_assets,
                    device=self.device,
                    sampling_strategy=discrete_config.get("sampling_strategy", "thompson")
                )
            else:
                raise ValueError(f"Unsupported discrete head type: {head_type}")
        
        # Confidence Head
        if head_configs.get("confidence", {}).get("enabled", False):
            confidence_config = head_configs["confidence"]
            head_type = confidence_config.get("type", "parametric")
            
            if head_type == "parametric":
                self.heads["confidence"] = ParametricConfidenceHead(
                    input_dim=backbone_output_dim,
                    hidden_dim=confidence_config.get("hidden_dim", 128),
                    n_assets=self.n_assets,
                    device=self.device
                )
            elif head_type == "bayesian":
                self.heads["confidence"] = BayesianConfidenceHead(
                    input_dim=backbone_output_dim,
                    hidden_dim=confidence_config.get("hidden_dim", 128),
                    n_assets=self.n_assets,
                    device=self.device,
                    sampling_strategy=confidence_config.get("sampling_strategy", "thompson")
                )
            else:
                raise ValueError(f"Unsupported confidence head type: {head_type}")
    
    def forward(self, observations: Dict[str, torch.Tensor], use_sampling: bool = False, temperature: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Process observations through the network.
        
        Args:
            observations: Dictionary of observation tensors for each processor
            use_sampling: Whether to use sampling for Bayesian heads (default: False)
            temperature: Dictionary of temperature values for each head
        Returns:
            Dictionary of outputs from each head:
            - For value head: {'value': tensor} of shape (batch_size, 1) or (1,) for single sample
            - For discrete head: {'action_probs': tensor} of shape (batch_size, n_assets, 3) or (n_assets, 3) for single sample
            - For confidence head: {'confidences': tensor} of shape (batch_size, n_assets) or (n_assets,) for single sample
        """
        # Check if observations dictionary has any keys (ONNX-friendly)
        has_observations = False
        for key in observations.keys():
            has_observations = True
            break
        
        if not has_observations:
            raise ValueError("No observations provided")
        
        if temperature is None:
            temperature = {}
            for key in self.heads.keys():
                temperature[key] = 1.0
            
        # Process each observation type through its processor
        processed_features = []
        for name, processor in self.processors.items():
            if name in observations:
                features = processor(observations[name])
                processed_features.append(features)
        
        # Concatenate all processed features
        if len(processed_features) == 0:
            raise ValueError("No observations provided for any processor")
        
        # Concatenate along the feature dimension
        combined_features = torch.cat(processed_features, dim=-1)  # Shape: (batch_size, total_features)
        
        # Process through backbone
        backbone_features = self.backbone(combined_features)  # Shape: (batch_size, hidden_dim)
        
        # Process through each head
        outputs = {}
        for name, head in self.heads.items():
            if name == "value":
                # For value head, process directly
                if isinstance(head, (BayesianValueHead)):
                    if use_sampling:
                        head_output = head.sample(backbone_features, temperature=temperature[name])
                    else:
                        head_output = head(backbone_features)
                else:
                    head_output = head(backbone_features)
                
                # Extract value from output
                if isinstance(head_output, dict):
                    if "value" in head_output:
                        outputs["value"] = head_output["value"]
                    elif "mean" in head_output:
                        outputs["value"] = head_output["mean"]
                    else:
                        raise ValueError(f"Unexpected dictionary output from value head: {head_output.keys()}")
                else:
                    outputs["value"] = head_output
            else:
                # For confidence head, process directly
                if name == "confidence":
                    # For confidence head
                    if isinstance(head, (BayesianConfidenceHead)):
                        if use_sampling:
                            head_output = head.sample(backbone_features, temperature=temperature[name])
                        else:
                            head_output = head(backbone_features)
                    else:
                        head_output = head(backbone_features)
                    
                    if isinstance(head_output, dict):
                        if "confidences" in head_output:
                            outputs["confidences"] = head_output["confidences"]
                        elif "alphas" in head_output and "betas" in head_output:
                            outputs["conf_alphas"] = head_output["alphas"]
                            outputs["conf_betas"] = head_output["betas"]
                        else:
                            raise ValueError(f"Unexpected dictionary output from confidence head: {head_output.keys()}")
                    else:
                        outputs["confidences"] = head_output
                else: # name == "discrete"
                    # For discrete head
                    if isinstance(head, (BayesianDiscreteHead)):
                        if use_sampling:
                            head_output = head.sample(backbone_features, temperature=temperature[name])
                        else:
                            head_output = head(backbone_features)
                    else:
                        head_output = head(backbone_features)
                    
                    if isinstance(head_output, dict):
                        if "action_probs" in head_output:
                            outputs["action_probs"] = head_output["action_probs"]
                        elif "alphas" in head_output and "betas" in head_output:
                            outputs["alphas"] = head_output["alphas"]
                            outputs["betas"] = head_output["betas"]
                        else:
                            raise ValueError(f"Unexpected dictionary output from discrete head: {head_output.keys()}")
                    else:
                        outputs["action_probs"] = head_output
        
        return outputs 