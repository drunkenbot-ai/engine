"""Training loop compatibility module.

The implementation lives in :mod:`training_impl` to keep modules focused.
"""
from .training_impl import train_model

__all__ = ["train_model"]

