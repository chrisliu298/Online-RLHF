from dataclasses import dataclass

from trl import DPOConfig


@dataclass
class DPOConfigWithAdditionalArgs(DPOConfig):
    reward_model_name: str = None
    nll_loss_alpha: float = 0.0
    choose_type: str = None
    num_generations: int = 8
    len_penalty: float = 0.0
