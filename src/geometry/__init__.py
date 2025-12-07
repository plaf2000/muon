"""
Geometric analysis utilities (Hessian, curvature, etc.).
"""

from .hessian import compute_lambda_max, hessian_vector_product
from .curvature import (
    compute_directional_curvature,
    compute_lambda_grad,
    compute_lambda_muon
)

__all__ = [
    "compute_lambda_max",
    "hessian_vector_product",
    "compute_directional_curvature",
    "compute_lambda_grad",
    "compute_lambda_muon"
]
