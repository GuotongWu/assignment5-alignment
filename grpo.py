import torch
import wandb
import pandas as pd
import numpy as np
import typer
from tqdm.auto import tqdm
from typing import Callable, Literal
from vllm import SamplingParams
from utils import init_vllm, init_wandb, build_prompt,  load_policy_into_vllm_instance
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from utils import tokenize_prompt_and_output, get_response_log_probs

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
    
    with torch.no_grad():
        clip_ratio = is_clipped.float().mean()
        log_ratio = policy_log_probs - old_log_probs
        approx_kl = 0.5 * torch.mean(log_ratio ** 2)
    return (
        loss, 
        {
            "is_clipped": is_clipped,
            "clip_ratio": clip_ratio,
            "approx_kl": approx_kl
        }
    )
    

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange) 
    return (loss, metadata)
    
    
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)
    
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
):
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    
    mean_loss = masked_mean(loss, response_mask, dim=None) / gradient_accumulation_steps
    mean_loss.backward()
    
    return mean_loss, metadata
    
def grpo_train_loop(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    train_device: str = "cuda:0",
    inference_device: str = "cuda:1",
    model_name: str = "Qwen3-0.6B",
    train_dataset_path: str = "data/sft-cs336-assign5-datasets/sft-reason/train.jsonl",
    seed: int = 42,
    cliprange: float = 1.0,
    warmup_ratio: float=0.03,
):
    args_dict = locals()
    init_wandb("GRPO", **args_dict)
    
    n_prompt_per_rollout_batch = rollout_batch_size // group_size
    train_df = pd.read_json(train_dataset_path)
    sampling_params = SamplingParams(
        n = group_size,
        temperature=sampling_temperature, top_p=1.0, min_tokens=sampling_min_tokens, max_tokens=sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output=True
    )
    tokenizer = AutoTokenizer.from_pretrained(f"model/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        f"model/{model_name}", torch_dtype=torch.bfloat16,
    ).to(train_device)
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    llm = init_vllm(model_id=f"model/{model_name}", device=inference_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization,
    )
    
    total_step = n_grpo_steps * epochs_per_rollout_batch * gradient_accumulation_steps
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=int(n_grpo_steps * epochs_per_rollout_batch * warmup_ratio)
    )
    
    reward_fn = r1_zero_reward_fn
    
    global_opt_step = 0
    progress_bar = tqdm(range(total_step))
    
    for step in range(n_grpo_steps):
        sample_data = train_df.sample(n=n_prompt_per_rollout_batch)
        
        raw_prompts = sample_data["problem"].to_list()
        prompts = build_prompt(raw_prompts)
        repeated_prompts = [pi for pi in prompts for _ in range(group_size)]
        
        ground_truths = sample_data["expected_answer"].to_list()
        repeated_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]
        
        load_policy_into_vllm_instance(model, llm)
        
        responses = llm.generate(prompts, sampling_params)
        rollout_responses = [output.text for ri in responses for output in ri.outputs]
        
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization)
        advantages, raw_rewards = advantages.to(train_device), raw_rewards.to(train_device)
        
        wandb.log(data={
            "rollout_step": step,
            "rollout/mean_rewards": metadata["mean_rewards"],
            "rollout/std_rewards": metadata["std_rewards"],
            "rollout/max_rewards": metadata["max_rewards"],
            "rollout/min_rewards": metadata["min_rewards"],
        }, step=step)
        
        tokenize_results = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer, device=train_device)
        
        input_ids = tokenize_results["input_ids"]
        labels = tokenize_results["labels"]
        response_mask = tokenize_results["response_mask"]
        
        
        micro_train_batch_size = train_batch_size // gradient_accumulation_steps
        
        old_log_probs = []
        
        with torch.no_grad():
            for i in range(gradient_accumulation_steps):
                micro_index = range(i*micro_train_batch_size, (i+1)*micro_train_batch_size)
                
                micro_input_ids, micro_labels, micro_response_mask, micro_raw_rewards, micro_advantages = get_microbatch_data(micro_index, input_ids, labels, response_mask, raw_rewards, advantages)
                
                old_log_probs.append(get_response_log_probs(model, micro_input_ids, micro_labels)["log_probs"])
            
        old_log_probs = torch.cat(old_log_probs, dim=0)
        
        for epoch in range(epochs_per_rollout_batch):
            for i in range(gradient_accumulation_steps):
                micro_index = range(i*micro_train_batch_size, (i+1)*micro_train_batch_size)
                
                micro_input_ids, micro_labels, micro_response_mask, micro_raw_rewards, micro_advantages = get_microbatch_data(micro_index, input_ids, labels, response_mask, raw_rewards, advantages)
                
                micro_policy_log_probs_entropy = get_response_log_probs(model, micro_input_ids, micro_labels, return_token_entropy=True)
                
                micro_policy_log_probs = micro_policy_log_probs_entropy["log_probs"]
                micro_token_entropy = micro_policy_log_probs_entropy["token_entropy"]                
                
                micro_old_log_probs = old_log_probs[micro_index, ...]
                
                mean_loss, metadata = grpo_microbatch_train_step(micro_policy_log_probs, micro_response_mask, gradient_accumulation_steps, loss_type, micro_raw_rewards, micro_advantages, micro_old_log_probs, cliprange=cliprange)
                
                wandb.log(data={
                    "train/token_entropy": micro_token_entropy,
                }, step=global_opt_step)
                
                if len(metadata) > 0:
                    wandb.log(data={
                        "train/mean_loss": mean_loss,
                        "train/clip_ratio": metadata["clip_ratio"],
                        "train/approx_kl": metadata["approx_kl"]
                    }, step=global_opt_step)
                    
                global_opt_step += 1
                progress_bar.update(n=1)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
       
       
       
def get_microbatch_data(micro_index: range, input_ids: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor, raw_rewards: torch.Tensor, advantages: torch.Tensor):
    micro_input_ids = input_ids[micro_index, ...]
    micro_labels = labels[micro_index, ...]
    micro_response_mask = response_mask[micro_index, ...]
    micro_raw_rewards = raw_rewards[micro_index, ...]
    micro_advantages = advantages[micro_index, ...]
    
    return micro_input_ids, micro_labels, micro_response_mask, micro_raw_rewards, micro_advantages
    
       

if __name__ == "__main__":
    typer.run(grpo_train_loop)