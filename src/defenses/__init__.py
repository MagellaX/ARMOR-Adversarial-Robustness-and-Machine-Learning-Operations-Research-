# Import defense implementations
from .adversarial_training import AdversarialDefense

__all__ = [
    'AdversarialDefense'
]

# Version info (optional)
__version__ = '0.1.0'

# Module docstring...
"""
This module provides implementations of various defense mechanisms against adversarial attacks.

Available Defenses:
    - AdversarialDefense: Implementation of adversarial training defense
"""