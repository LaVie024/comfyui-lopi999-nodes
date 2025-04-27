from .lopi999_utils import RandomSDXLLatentSize
from .lopi999_utils import RandomNormalDistribution
from .lopi999_utils import AdvancedTextSwitch
from .lopi999_utils import ZipfSchedulerNode
from .lopi999_utils import ZetaSchedulerNode
from .lopi999_utils import SDXLEmptyLatentSizePicker_v2
from .lopi999_utils import Lopi999InputParameters
from .lopi999_utils import ModelParameters

NODE_CLASS_MAPPINGS = {
    "RandomSDXLLatentSize": RandomSDXLLatentSize,
    "RandomNormalDistribution": RandomNormalDistribution,
    "AdvancedTextSwitch": AdvancedTextSwitch,
    "ZipfSchedulerNode": ZipfSchedulerNode,
    "ZetaSchedulerNode": ZetaSchedulerNode,
    "SDXLEmptyLatentSizePicker_v2": SDXLEmptyLatentSizePicker_v2,
    "Lopi999InputParameters": Lopi999InputParameters,
    "ModelParameters": ModelParameters
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomSDXLLatentSize": "Random SDXL Latent Size",
    "RandomNormalDistribution": "Random Normal Distribution",
    "AdvancedTextSwitch": "Advanced Text Switch",
    "ZipfSchedulerNode": "Zipf Scheduler",
    "ZetaSchedulerNode": "Zeta Scheduler",
    "SDXLEmptyLatentSizePicker_v2": "SDXL Empty Latent Size Picker v2",
    "Lopi999InputParameters": "Input Parameters (lopi999)",
    "ModelParameters": "Model Parameters"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
