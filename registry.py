"""
Architecture registry.

Imports each architecture module and exposes MODEL_CONFIGS, a dict mapping
the CLI model name to its ArchitectureConfig instance.

To add a new architecture:
  1. Create quant_probe/models/<name>.py exporting a CONFIG instance.
  2. Add one import line and one entry in MODEL_CONFIGS below.
"""

from quant_probe.models import wan, zimage
from quant_probe.models.base import ArchitectureConfig
from typing import Dict

MODEL_CONFIGS: Dict[str, ArchitectureConfig] = {
    wan.CONFIG.name:    wan.CONFIG,
    zimage.CONFIG.name: zimage.CONFIG,
}
