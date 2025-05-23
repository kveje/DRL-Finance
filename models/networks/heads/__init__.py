"""Heads for trading agents."""

from models.networks.heads.base_head import BaseHead
from models.networks.heads.value_head import ParametricValueHead, BayesianValueHead
from models.networks.heads.discrete_head import ParametricDiscreteHead, BayesianDiscreteHead
from models.networks.heads.confidence_head import ParametricConfidenceHead, BayesianConfidenceHead


__all__ = ["BaseHead", "ParametricValueHead", "BayesianValueHead", "ParametricDiscreteHead", "BayesianDiscreteHead", "ParametricConfidenceHead", "BayesianConfidenceHead"]

