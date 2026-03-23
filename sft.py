import torch
from transformers import PreTrainedTokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict:
    
    return {
        "input_ids": None,
        "labels": None,
        "response_mask": None
    }
    
if __name__ == "__main__":
    pass