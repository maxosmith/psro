"""Empirical games."""
from .normal_form import NormalForm
from .normal_form_sql import NormalForm as NormalFormSQL

__all__ = (
    "NormalForm",
    "NormalFormSQL",
)
