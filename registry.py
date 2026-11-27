"""
Architecture registry.

Imports each architecture module and exposes MODEL_CONFIGS, a dict mapping
the CLI model name to its ArchitectureConfig instance.

To add a new architecture:
  1. Create quant_probe/models/<name>.py exporting a CONFIG instance.
  2. Add one import line and one entry in MODEL_CONFIGS below.
"""

from quant_probe.models import wan21, wan22, zimage, ernie, qwen21, moss

from quant_probe.models.base import ArchitectureConfig
from typing import Dict

MODEL_CONFIGS: Dict[str, ArchitectureConfig] = {
    wan21.CONFIG.name: wan21.CONFIG,
    wan22.CONFIG.name: wan22.CONFIG,
    zimage.CONFIG.name: zimage.CONFIG,
    ernie.CONFIG.name: ernie.CONFIG,
    qwen21.CONFIG.name: qwen21.CONFIG,
    moss.CONFIG.name: moss.CONFIG,
}
