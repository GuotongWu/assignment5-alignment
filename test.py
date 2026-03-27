import json
import regex as re
import pandas as pd
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

EXTRACTED_TEMPLATE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

problem = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: For what value of $x$ is the following equation true: $6500 + x - 4500 = 3400 + 2000$?\nAssistant: <think>"
response = " reasoning process here\n</think> <answer> 2000 </answer>"

match = EXTRACTED_TEMPLATE.search(response)
print(match)
print(r1_zero_reward_fn(" reasoning process here\n</think> <answer> 2000 </answer>", "2000"))