"""Interface identification for polygon-based spatial data.

Re-exports from the shared ``_interface`` module.
"""

from spatioloji_s.spatial._interface import (
    _compute_tortuosity,
    _empty_result,
    _validate_inputs,
    filter_interface,
    identify_interface,
)

__all__ = ["identify_interface", "filter_interface", "_validate_inputs", "_empty_result", "_compute_tortuosity"]
