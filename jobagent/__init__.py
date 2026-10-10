"""Agentic components for the AI job agent.

This package sits beside ``app/`` and ``automation/`` and does not change
their behaviour.
"""

import os

# Must be set before sentence_transformers is imported anywhere in the
# package. The embedding model is PyTorch; without this, transformers also
# tries to load TensorFlow and fails when Keras 3 is installed without
# tf-keras.
os.environ.setdefault("USE_TF", "0")
