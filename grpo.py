import torch
import pandas as pd
import numpy as np
from typing import Callable, Literal

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        raw_rewards.append(reward_fn(response, ground_truth))
    raw_rewards = pd.DataFrame(raw_rewards)["reward"].to_numpy().reshape(-1, group_size)
    mean_rewards = np.mean(raw_rewards, -1, keepdims=True)
    advantages = (raw_rewards - mean_rewards)
    
    if normalize_by_std:
        std_rewards = np.std(raw_rewards, -1, keepdims=True, ddof=1)
        advantages /= std_rewards + advantage_eps
    
    return (
        torch.tensor(advantages.flatten()),
        torch.tensor(raw_rewards.flatten()),
        {
            "mean_rewards": raw_rewards.mean().item(),
            "std_rewards": raw_rewards.std().item(),
            "max_rewards": raw_rewards.max().item(),
            "min_rewards": raw_rewards.min().item(),
        }
    )
    

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs
    

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio_log_probs = torch.exp(policy_log_probs - old_log_probs)
    lhs = ratio_log_probs * advantages
    rhs = torch.clip(ratio_log_probs, 1 - cliprange, 1 + cliprange) * advantages
    
    is_clipped = rhs < lhs
    loss = -torch.min(lhs, rhs)
    return (loss, {"is_clipped": is_clipped})
    

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
):
    pass