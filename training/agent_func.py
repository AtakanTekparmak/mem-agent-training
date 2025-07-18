from typing import Dict, Any
import os
import json

from agent.utils import extract_reply, extract_python_code, extract_thoughts, format_results, count_block_occurrences
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

    # Apply stop token logic - truncate action after closing tags
    # Priority: </python> first, then </reply>
    if "</python>" in action:
        python_end_idx = action.find("</python>") + len("</python>")
        action = action[:python_end_idx]
    elif "</reply>" in action:
        reply_end_idx = action.find("</reply>") + len("</reply>")
        action = action[:reply_end_idx]

    # Count occurrences of each block type
    think_count = count_block_occurrences(action, "<think>", "</think>")
    python_count = count_block_occurrences(action, "<python>", "</python>")
    reply_count = count_block_occurrences(action, "<reply>", "</reply>")

    # Extract the content from blocks
    python_code = extract_python_code(action)
    reply = extract_reply(action)
    thoughts = extract_thoughts(action)

    # Check if the action contains a python code, reply, or thoughts block with content
    python_code_exists = len(python_code.strip()) > 0
    reply_exists = len(reply.strip()) > 0
    thoughts_exists_and_long_enough = len(thoughts.strip()) > THOUGHTS_MIN_LENGTH

    # Initialize the reward and done flag
    reward = 0.0
    done = False
    error_msg = None

    # Check format requirements
    format_errors = []
    
    # Check for mandatory think block
    if think_count == 0:
        format_errors.append("Missing mandatory <think> block")
    elif think_count > 1:
        format_errors.append(f"Multiple <think> blocks found ({think_count}), only one allowed")
    elif not thoughts_exists_and_long_enough:
        format_errors.append(f"<think> block is too short (minimum {THOUGHTS_MIN_LENGTH} characters)")
    
    # Check for required python or reply block
    if python_count == 0 and reply_count == 0:
        format_errors.append("Missing either <python> or <reply> block")
    
    # If there are format errors, penalize and return early
    if format_errors:
        error_msg = " ".join(format_errors)
        next_observation = (
            observation + action + 
            f"\n [ERROR] Format violation: {error_msg}" +
            "\n<assistant>"
        )
        reward = -0.2  # Penalty for format violations
    elif think_count == 1 and thoughts_exists_and_long_enough:
        # Good format with proper think block
        reward = 0.1
        
        if python_code_exists and python_count == 1:
            local_vars, exec_error = execute_sandboxed_code(
                code=python_code,
                allowed_path=MEMORY_PATH,
                import_module="agent.tools",
            )

            next_observation = (
                observation + action +
                format_results(local_vars, exec_error) +
                ("\n<assistant>")
            )

            # Additional reward for valid python block
            reward += 0.1
        elif reply_exists and reply_count == 1:
            question = extract_question(observation)
            reward += max(0.1, get_reward(question, reply, label))
            done = True

            next_observation = (
                observation + action + "\n</s>"
            )
        else:
            # Handle case where blocks exist but are empty
            if python_count == 1:
                error_msg = "Python block is empty"
            elif reply_count == 1:
                error_msg = "Reply block is empty"
            else:
                error_msg = "Unknown error - valid format but no executable content"
            
            next_observation = (
                observation + action + 
                f"\n [ERROR] {error_msg}" +
                "\n<assistant>"
            )
            reward = -0.2  # Penalty for empty blocks
    else:
        # Fallback case for any other unexpected scenarios
        next_observation = (
            observation + action + 
            "\n [ERROR] Unexpected format issue" +
            "\n<assistant>"
        )
        reward = -0.2

    step_idx += 1

    # If reward is higher than 1, set it to 1
    if reward > 0.0:
        reward = min(reward, 1.0)
    else:
        reward = max(reward, -1.0)
    reward = torch.tensor(reward)

    # Set vLLM sampling params with stop tokens
    # Priority: </python> first, then </reply> (same as our makeshift implementation)
    sampling_params = kwargs.get("sampling_params", {})
    if sampling_params is None:
        sampling_params = {}
    
    # Add stop tokens for vLLM native support
    sampling_params["stop"] = ["</python>", "</reply>"]

    return {
        "rewards": reward,
        "scores": reward,
        "next_observation": next_observation,
        "done": done,
        "sampling_params": sampling_params,
        "extra_logs": {},
    }
    