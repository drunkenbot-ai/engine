"""Backward-compatible facade for training APIs.

Implementation is split into focused modules while legacy imports remain valid.
"""
from .training_core import *
from .training_runtime import *
from .training_evaluation import *
from .training_resume import *
from .training_loop import *
from .training_checkpoint import *

