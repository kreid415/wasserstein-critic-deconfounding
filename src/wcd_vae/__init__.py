"""wcd_vae package.

Back-compatibility shims: the modules ``wcd_vae.data``, ``wcd_vae.hyperparameter``,
``wcd_vae.metrics`` and ``wcd_vae.scCRAFT.model`` were relocated during the
authored-vs-upstream refactor. The names below re-export from their new homes so
existing scripts and notebooks keep importing the old paths. New code should import
from ``wcd_vae.wcd`` (authored) or ``wcd_vae.scCRAFT`` (upstream backbone).
"""
import sys as _sys

from wcd_vae.wcd import data as _data
from wcd_vae.wcd import evaluation as _evaluation
from wcd_vae.wcd import hyperparameter as _hyperparameter
from wcd_vae.wcd import training as _training

# Map legacy dotted module paths to the relocated modules.
_sys.modules.setdefault("wcd_vae.data", _data)
_sys.modules.setdefault("wcd_vae.metrics", _evaluation)
_sys.modules.setdefault("wcd_vae.hyperparameter", _hyperparameter)
_sys.modules.setdefault("wcd_vae.scCRAFT.model", _training)
