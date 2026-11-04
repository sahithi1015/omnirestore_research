"""Canonical backbone. Shift is explicit; legacy scripts default to non-shifted.

The checkpoint does not record the shift setting. Both modes share state keys.
"""
from .wadt_net_shift import WADNet
