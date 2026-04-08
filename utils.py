import os
import time
import torch
import wandb
import json
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from vllm import LLM, SamplingParams
from unittest.mock import patch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from typing import Callable
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"

class MathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, select_num: int = -1):
        if select_num != -1:
            self.df = df.head(select_num)
        else:
            self.df = df
        
        with open(PROMPT_TEMPLATE_PATH, "r") as f:
            self.template = f.read()

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return self.template.format(question=row["problem"]), row["reasoning_trace"]

def build_prompt(raw_prompts):
    with open(PROMPT_TEMPLATE_PATH, "r") as f:
        template = f.read()
    return [template.format(question=pi) for pi in raw_prompts]
  
def init_wandb(group_type, **kwargs):
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    
    current_time = time.strftime("%m%d_%H%M")
    
    if group_type == "SFT":
        wandb.init(
            project="assignment5-alignment",
            group="SFT",
            name=f"sft-{kwargs['model_name']}-lr{kwargs['learning_rate']:.4e}-bs{kwargs['batch_size']}-{current_time}",
            config=kwargs
        )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
    elif group_type == "GRPO":
        wandb.init(
            project="assignment5-alignment",
            group="GRPO",
            name=f"grpo-{kwargs['model_name']}-lr{kwargs['learning_rate']:.4e}-{kwargs['loss_type']}-bs{kwargs['train_batch_size']}-{current_time}",
            config=kwargs
        )
        wandb.define_metric("rollout_step")
        wandb.define_metric("train_step")
        
        wandb.define_metric("rollout/*", step_metric="rollout_step")
        wandb.define_metric("train/*", step_metric="train_step")
    
def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.7
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
        )
    
@torch.no_grad()
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    
def evaluate_metrics(records: list[dict], eval_step: int):
    df = pd.json_normalize(records)
    df = df.rename(columns={
        'reward.format_reward': "format_reward",
        'reward.answer_reward':"answer_reward",
        'reward.reward': "reward"})
    df_form1_ans1 = df[(df["format_reward"] == 1) & (df["answer_reward"] == 1)]
    df_form1_ans0 = df[(df["format_reward"] == 1) & (df["answer_reward"] == 0)]
    df_form0_ans0 = df[(df["format_reward"] == 0) & (df["answer_reward"] == 0)]
    
    wandb.log({
        "eval_step": eval_step,
        "eval/mean_reward": df["reward"].mean(),
        "eval/form1_ans1": len(df_form1_ans1) / len(df),
        "eval/form1_ans0": len(df_form1_ans0) / len(df),
        "eval/form0_ans0": len(df_form0_ans0) / len(df),
    })


def evaluate_vllm(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str | int | float], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str | int | float],
    eval_sampling_params: SamplingParams,
    eval_step: int,
    args = None,
    is_end: bool = False,
):
    responses = vllm_model.generate(prompts, eval_sampling_params)
    records = []
    for response, ground_truth, prompt in zip(responses, ground_truths, prompts):
        response_text = response.outputs[0].text
        reward = reward_fn(response_text, ground_truth)
        records.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "response": response_text,
            "reward": reward
        })
        
    evaluate_metrics(records, eval_step)

    if is_end:
        current_time = time.strftime("%m%d_%H%M")
        with open(f"output/sft/{args.model_name}-lr{args.learning_rate:.4e}-bs{args.batch_size}-{current_time}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
            

def compute_entropy(logits: torch.Tensor)->torch.Tensor:
    new_logits = torch.softmax(logits, dim=-1)
    logsumexp_logits = torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(new_logits * (logits - logsumexp_logits), dim=-1)
       
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
    device: str
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs, padding=False, truncation=False)["input_ids"]
    output_ids = tokenizer(output_strs, padding=False, truncation=False)["input_ids"]
    
    concat_ids = []
    len_concat_ids = []
    len_input_ids = []
    for pi, oi in zip(prompt_ids, output_ids):
        concat_ids.append(torch.tensor(pi + oi, dtype=torch.long))
        len_input_ids.append(len(pi))
        len_concat_ids.append(len(pi) + len(oi))
    
    concat_ids = pad_sequence(concat_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right').to(device)
    len_concat_ids = torch.tensor(len_concat_ids, dtype=torch.int32).to(device)
    len_input_ids = torch.tensor(len_input_ids, dtype=torch.int32).to(device)
    
    raw_mask = torch.arange(concat_ids.shape[1]).to(device)
    response_mask = ((len_input_ids.unsqueeze(dim=-1) <= raw_mask) & ( raw_mask < len_concat_ids.unsqueeze(dim=-1)))
    
    return {
        "input_ids": concat_ids[..., :-1],
        "labels": concat_ids[..., 1:],
        "response_mask": response_mask[...,1:]
    }
    
    
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
)->dict[str, torch.Tensor]:
    token_entropy = None
    logits = model(input_ids).logits
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
    new_logits = F.log_softmax(logits, dim=-1)
    return {
        "log_probs": torch.gather(new_logits, dim=-1, index=labels.unsqueeze(dim=-1)).squeeze(dim=-1),
        "token_entropy": token_entropy
    }