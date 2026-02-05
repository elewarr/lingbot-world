"""PyTorch backend implementation.

This module wraps the existing WanModel for use with the backend abstraction.
It provides the default backend that maintains full backward compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .base import Backend, register_backend


@register_backend('pytorch')
class PyTorchBackend(Backend):
    """PyTorch backend wrapping the existing WanModel.
    
    This backend uses the original PyTorch implementation and is the
    default for maximum compatibility. It supports CUDA, MPS, and CPU.
    """
    
    @property
    def name(self) -> str:
        return 'pytorch'
    
    @property
    def is_available(self) -> bool:
        """PyTorch is always available (required dependency)."""
        return True
    
    def load_model(
        self,
        checkpoint_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Load WanModel from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            model_config: Optional config overrides (subfolder, etc.)
            device: Target device ('cuda', 'mps', 'cpu')
            dtype: Model dtype (torch.float16, torch.bfloat16, etc.)
            
        Returns:
            WanModel instance with loaded weights
        """
        # Lazy import to avoid circular dependencies
        from wan.modules.model import WanModel
        
        config = model_config or {}
        subfolder = config.get('subfolder', None)
        
        # Determine dtype
        if dtype is None:
            # Default based on device
            if device == 'mps':
                dtype = torch.float16
            elif device == 'cuda':
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        
        logging.info(f"PyTorchBackend: Loading model from {checkpoint_path}")
        if subfolder:
            logging.info(f"  subfolder: {subfolder}")
        logging.info(f"  dtype: {dtype}, device: {device}")
        
        model = WanModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        )
        
        # Set to eval mode
        model.eval()
        model.requires_grad_(False)
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        return model
    
    def forward(
        self,
        model: Any,
        x: Any,
        t: Any,
        context: Any,
        seq_len: int,
        y: Optional[Any] = None,
        dit_cond_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute WanModel forward pass.
        
        Args:
            model: WanModel instance
            x: List of input video tensors [C, F, H, W]
            t: Timestep tensor [B]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length
            y: Optional conditional input for i2v
            dit_cond_dict: Camera conditioning dict
            
        Returns:
            List of denoised video tensors
        """
        return model(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            y=y,
            dit_cond_dict=dit_cond_dict,
        )
    
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert PyTorch tensor to numpy array.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            NumPy array
        """
        return tensor.detach().cpu().numpy()
    
    def from_numpy(
        self,
        array: np.ndarray,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Convert numpy array to PyTorch tensor.
        
        Args:
            array: NumPy array
            device: Target device
            dtype: Target dtype
            
        Returns:
            PyTorch tensor
        """
        tensor = torch.from_numpy(array)
        
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    def move_to_device(self, model: Any, device: str) -> Any:
        """Move model to specified device.
        
        Args:
            model: WanModel instance
            device: Target device string
            
        Returns:
            Model on target device
        """
        return model.to(device)
    
    def set_eval_mode(self, model: Any) -> Any:
        """Set model to evaluation mode.
        
        Args:
            model: WanModel instance
            
        Returns:
            Model in eval mode
        """
        model.eval()
        model.requires_grad_(False)
        return model
    
    def get_model_device(self, model: Any) -> torch.device:
        """Get the device where model weights reside.
        
        Args:
            model: WanModel instance
            
        Returns:
            torch.device of model parameters
        """
        return next(model.parameters()).device
    
    def synchronize(self) -> None:
        """Synchronize device (wait for all operations to complete)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
