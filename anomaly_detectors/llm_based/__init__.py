"""
LLM-based Anomaly Detection Module

This module provides LLM-enhanced anomaly detection capabilities using language modeling
for probability-based anomaly detection.

Features:
- Language model-based anomaly detection
- Probability-based sequence scoring
- Dynamic-aware encoding for temporal patterns
- Prototype-based reprogramming
- Enhanced explanations and context awareness
"""

from .llm_anomaly_detector import LLMAnomalyDetector

__all__ = [
    'LLMAnomalyDetector'
] 