import json
import regex as re
import pandas as pd
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import List, Callable

MODEL_NAME = "Qwen3-0.6B"
VALID_DATASET_PATH = "data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"
PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"
EXTRACTED_TEMPLATE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    
def build_prompt(questions):
    with open(PROMPT_TEMPLATE_PATH, "r") as f:
        template = f.read()
    return [template.format(question=question) for question in questions]

def evaluate_vllm(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str | int | float], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str | int | float],
    eval_sampling_params: SamplingParams
):
    responses = vllm_model.generate(prompts, eval_sampling_params)
    records = []
    for response, ground_truth, prompt in zip(responses, ground_truths, prompts):
        match = EXTRACTED_TEMPLATE.search(response.outputs[0].text)
        if match:
            extracted_answer = match.group(1).strip()
        else:
            extracted_answer = ""
            
        reward = reward_fn(extracted_answer, ground_truth)
        records.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "response": response.outputs[0].text,
            "reward": reward
        })
    with open(f"output/baseline/{MODEL_NAME}_pre20.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
        
def main():
    valid_dataset = pd.read_json(VALID_DATASET_PATH)
    prompts = valid_dataset["problem"].to_list()[:20]
    ground_truths = valid_dataset["expected_answer"].to_list()[:20]
    
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True)
    
    llm = LLM(model=f"model/{MODEL_NAME}", max_num_batched_tokens=2048, max_model_len=2048)
    
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)

if __name__ == "__main__":
    main()