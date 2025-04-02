from .lopi999_utils import RandomSDXLLatentSize
from .lopi999_utils import RandomNormalDistribution
from .lopi999_utils import AdvancedTextSwitch
from .lopi999_utils import ZipfSchedulerNode

NODE_CLASS_MAPPINGS = {
    "RandomSDXLLatentSize": RandomSDXLLatentSize,
    "RandomNormalDistribution": RandomNormalDistribution,
    "AdvancedTextSwitch": AdvancedTextSwitch,
    "ZipfSchedulerNode": ZipfSchedulerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomSDXLLatentSize": "Random SDXL Latent Size",
    "RandomNormalDistribution": "Random Normal Distribution",
    "AdvancedTextSwitch": "Advanced Text Switch",
    "ZipfSchedulerNode": "Zipf Scheduler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
