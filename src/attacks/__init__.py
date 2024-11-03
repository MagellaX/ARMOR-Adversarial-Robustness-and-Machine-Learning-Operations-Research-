# Import attack implementations
from .evasion import EvasionAttacks
from .poisoning import PoisoningAttacks

# Define what gets exported
__all__ = [
    'EvasionAttacks',
    'PoisoningAttacks'
]

# Version info (optional)
__version__ = '0.1.0'

# Module docstring
"""
This module provides implementations of various adversarial attacks.

Available Attacks:
    - EvasionAttacks: Implementation of FGSM and PGD attacks
    - PoisoningAttacks: Implementation of backdoor and label flipping attacks
"""