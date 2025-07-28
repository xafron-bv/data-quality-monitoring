"""
GPU Utility Functions

Shared utilities for GPU detection and device management across ML modules.
"""

import torch


def get_optimal_device(use_gpu: bool = True) -> str:
    """
    Determine the optimal device to use for ML operations.

    Args:
        use_gpu: Whether to use GPU if available

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if not use_gpu:
        return 'cpu'

    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def print_device_info(device: str, context: str = ""):
    """
    Print information about the selected device.

    Args:
        device: The device string
        context: Optional context string to include in the message
    """
    context_str = f" for {context}" if context else ""

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        print(f"Using NVIDIA GPU ({gpu_name}){context_str}")
    elif device == 'mps':
        print(f"Using Apple M1/M2 GPU (MPS){context_str}")
    else:
        print(f"Using CPU{context_str}")


def is_gpu_device(device: str) -> bool:
    """
    Check if the device is a GPU (CUDA or MPS).

    Args:
        device: Device string to check

    Returns:
        True if device is GPU, False otherwise
    """
    return device in ('cuda', 'mps')


def get_optimal_batch_size(device: str, default_gpu: int = 40960, default_cpu: int = 5120) -> int:
  """
  Get optimal batch size based on device type.

  Args:
    device: Device string
    default_gpu: Default batch size for GPU
    default_cpu: Default batch size for CPU

  Returns:
    Optimal batch size
  """
  return default_gpu if is_gpu_device(device) else default_cpu
