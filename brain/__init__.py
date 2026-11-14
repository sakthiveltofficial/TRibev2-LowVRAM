"""
Brain prediction head package for TRIBE-inspired multimodal brain encoder.

Modules:
    subject_layers: LightweightSubjectHead, ROIHead, BrainPredictionHead
    losses:         BrainLoss (MSE + Pearson composite)
    metrics:        OnlinePearsonCorr, BrainMetrics
"""

from .subject_layers import LightweightSubjectHead, ROIHead, BrainPredictionHead
from .losses import BrainLoss
from .metrics import OnlinePearsonCorr, BrainMetrics

__all__ = [
    "LightweightSubjectHead",
    "ROIHead",
    "BrainPredictionHead",
    "BrainLoss",
    "OnlinePearsonCorr",
    "BrainMetrics",
]
