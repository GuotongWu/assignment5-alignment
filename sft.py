import torch
import wandb
import argparse
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers import AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from vllm import SamplingParams
from utils import MathDataset, init_vllm, init_wandb, build_prompt, evaluate_vllm, load_policy_into_vllm_instance
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

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
        concat_ids.append(torch.tensor(pi + oi, dtype=torch.int32))
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
    
def compute_entropy(logits: torch.Tensor)->torch.Tensor:
    new_logits = torch.softmax(logits, dim=-1)
    logsumexp_logits = torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(new_logits * (logits - logsumexp_logits), dim=-1)

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

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None
) -> torch.Tensor:
    masked_tensor = tensor * mask.to(tensor.dtype)
    return torch.sum(masked_tensor, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sum_masked_probs = masked_normalize(
        policy_log_probs, 
        response_mask, 
        normalize_constant,
        dim=1
    )
    loss = -sum_masked_probs.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}


def train(args):
    train_df = pd.read_json(args.train_dataset_path, lines=True)
    train_dataset = MathDataset(train_df, args.select_num)
    train_dataloder = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    
    eval_df = pd.read_json(args.eval_dataset_path, lines=True)
    eval_prompts = build_prompt(eval_df["problem"].to_list())
    
    tokenizer = AutoTokenizer.from_pretrained(f"model/{args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"model/{args.model_name}").to(args.train_device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataset) * args.epoch_num // (args.batch_size * args.gradient_accumulation_steps)
    sheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    step = 0
    
    llm = init_vllm(f"model/{args.model_name}", device=args.eval_device, seed=42, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    
    accumulated_loss = 0.0
    
    for _ in range(args.epoch_num):
        for idx, (prompt_strs, output_strs) in enumerate(train_dataloder):
            tokenize_results = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer, args.train_device)
            
            input_ids = tokenize_results["input_ids"]
            labels = tokenize_results["labels"]
            response_mask = tokenize_results["response_mask"]
            
            log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
            loss, _ = sft_microbatch_train_step(log_probs, response_mask, args.gradient_accumulation_steps)
            
            accumulated_loss += loss.item()
            
            if (idx + 1) % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                sheduler.step()
                optimizer.zero_grad()
                
                current_lr = sheduler.get_last_lr()[0]
                step += 1
                
                wandb.log({
                    "train_step": step, 
                    "train/loss": accumulated_loss, 
                    "train/lr": current_lr})
                
                accumulated_loss = 0.0
                
                if step % int(total_steps * args.eval_interval_ratio + 1) == 0:
                    load_policy_into_vllm_instance(model, llm)
                    evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_df["expected_answer"].to_list(), sampling_params, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-0.6B")
    parser.add_argument("--train_dataset_path", type=str, default="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, default="data/sft-cs336-assign5-datasets/sft-reason/val.jsonl")
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--eval_device", type=str, default="cuda:1")
    parser.add_argument("--select_num", type=int, default=128)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_interval_ratio", type=float, default=0.05)
    
    args = parser.parse_args()
    
    init_wandb(args)
    train(args)