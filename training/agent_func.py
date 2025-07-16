from typing import Dict, Any
import os
import json

from agent.utils import extract_reply, extract_python_code, extract_thoughts, format_results
from agent.engine import execute_sandboxed_code

from training import MEMORY_PATH
from training.reward import get_reward

import torch

# Load hyperparameters
try:
    with open("config.json", "r") as f:
        config = json.load(f)
        THOUGHTS_MIN_LENGTH = config["hyperparameters"]["thoughts_min_length"]
except:
    raise ValueError("config.json not found or the thoughts_min_length key is not present in hyperparameters")

# Global states for the environment
step_idx = 0
max_steps = 10

def extract_question(observation: str) -> str:
    """
    Extract the question from the observation.

    Args:
        observation: The input prompt/expression

    Returns:
        str: The question
    """
    if "<|im_start|>user" in observation:
        if "<|im_start|>assistant" in observation:
            extracted_question = observation.split("<|im_start|>user")[1].split("<|im_start|>assistant")[0].strip()
            if "<|im_end|>" in extracted_question:
                return extracted_question.split("<|im_end|>")[0].strip()
            else:
                return extracted_question
        else:
            raise ValueError("Trying to get question from observation but no assistant block found")
    else:
        raise ValueError(f"Observation does not contain a question")

async def step(observation, action, label, **kwargs) -> Dict[str, Any]:
    """
    Step function for the agent.

    Args:
        observation: The input prompt/expression
        action: The language model's response
        label: Agent identifier or additional information

    Returns:
        Dict[str, Any]: A dictionary containing:
            - rewards: Reward value for advantage calculation
            - scores: Reward value for dynamic filtering (same value as rewards for our case)
            - next_observation: The updated observation after the step
            - done: Boolean indicating if the episode is complete
            - sampling_params: Parameters for vLLM sampling
            - extra_logs: Additional logging information in dictionary format
    """
    global step_idx, max_steps
    print(f"step_idx: {step_idx}, max_steps: {max_steps}")

    if step_idx >= max_steps:
        done = True
        next_observation = (
            observation + action +
            "\n [WARNING] You have reached the maximum number of steps."
        )
        return {
            "rewards": torch.tensor(0),
            "scores": torch.tensor(0),
            "next_observation": next_observation,
            "done": done,
            "sampling_params": kwargs.get("sampling_params", None),
            "extra_logs": {},
        }

    # Extract the python code and reply
    python_code = extract_python_code(action)
    reply = extract_reply(action)
    thoughts = extract_thoughts(action)

    # Check if the action contains a python code, reply, or thoughts block
    python_code_exists = len(python_code.strip()) > 0
    reply_exists = len(reply.strip()) > 0
    thoughts_exists_and_long_enough = len(thoughts.strip()) > THOUGHTS_MIN_LENGTH

    # Initialize the reward and done flag
    reward = 0.1 if thoughts_exists_and_long_enough else 0.0
    done = False

    if python_code_exists and reply_exists:
        next_observation = (
            observation + action + 
            "\n [ERROR] You cannot provide a <python> and a <reply> block at the same time." +
            "\n<assistant>"
        )
    elif python_code_exists:
        local_vars, error_msg = execute_sandboxed_code(
            code=python_code,
            allowed_path=MEMORY_PATH,
            import_module="agent.tools",
        )

        next_observation   = (
            observation + action +
            format_results(local_vars, error_msg) +
            ("\n<assistant>")
        )

        # format reward 
        reward += 0.1
    elif reply_exists:
        question = extract_question(observation)
        reward += max(0.1, get_reward(question, reply, label))
        done = True

        next_observation = (
            observation + action + "\n</s>"
        )
    else:
        # Handle the case where the model output didn't contain a recognised
        # block so that `next_observation` is always defined.
        next_observation = (
            observation + action +
            "\n [ERROR] Missing <python> or <reply> block." +
            "\n<assistant>"
        )

    step_idx += 1

    # If reward is higher than 1, set it to 1
    reward = min(reward, 1.0)
    reward = torch.tensor(reward)

    return {
        "rewards": reward,
        "scores": reward,
        "next_observation": next_observation,
        "done": done,
        "sampling_params": kwargs.get("sampling_params", None),
        "extra_logs": {},
    }
    