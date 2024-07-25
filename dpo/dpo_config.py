from dataclasses import dataclass

from trl import DPOConfig


@dataclass
class DPOConfigWithAdditionalArgs(DPOConfig):
    nll_loss_alpha: float = 0.0
    choose_type: str = None
