"""
Package initialisation for the Lumina NAG integration.  Expose the
classes defined in model.py so they can be imported directly from
comfyui_nag.lumina.
"""

from .model import NAGNextDiT, NAGNextDiTSwitch  # noqa: F401

__all__ = [
    "NAGNextDiT",
    "NAGNextDiTSwitch",
]