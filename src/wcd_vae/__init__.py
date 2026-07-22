"""wcd_vae package.

All code is authored (K. Reid). Back-compatibility shims re-export the relocated
modules ``wcd_vae.data``, ``wcd_vae.hyperparameter``, and ``wcd_vae.metrics`` from their
homes under ``wcd_vae.wcd`` so existing scripts keep importing the old paths. New code
should import from ``wcd_vae.wcd``.
"""
import sys as _sys

from wcd_vae.wcd import data as _data
from wcd_vae.wcd import evaluation as _evaluation
from wcd_vae.wcd import hyperparameter as _hyperparameter

# Map legacy dotted module paths to the relocated modules.
_sys.modules.setdefault("wcd_vae.data", _data)
_sys.modules.setdefault("wcd_vae.metrics", _evaluation)
_sys.modules.setdefault("wcd_vae.hyperparameter", _hyperparameter)
