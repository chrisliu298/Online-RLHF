from dataclasses import dataclass

from trl import DPOConfig


@dataclass
class DPOConfigWithAdditionalArgs(DPOConfig):
    choose_type: str = None
    num_generations: int = None
    nll_loss_alpha: float = 0.0
    len_penalty: float = 0.0
    use_prev_iter_as_ref: str = "false"
    reward_model_name: str = None
    loss_type: str = "sigmoid"
