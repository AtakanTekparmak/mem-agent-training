from typing import Dict, Any
import json

from agent.utils import extract_reply, extract_python_code, extract_thoughts

from training.action_processor import process_action_base
from training.retrieval import calculate_retrieval_reply_reward
from training.update import calculate_update_reply_reward
from training.utils import Task, TaskType, extract_task_from_label, format_agent_response, remove_all_thinks_except_last

import torch
from vllm import SamplingParams

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
    
    # Remove all the <think> blocks except the last one
    observation = remove_all_thinks_except_last(observation)
    
    # Truncate the action after the closing tags
    if "</python>" in action:
        python_end_idx = action.find("</python>") + len("</python>")
        action = action[:python_end_idx]

    if "</reply>" in action:
        reply_end_idx = action.find("</reply>") + len("</reply>")
        action = action[:reply_end_idx]

    # Extract the python code and reply
    python_code = extract_python_code(action)
    reply = extract_reply(action)
    thoughts = extract_thoughts(action)

    # Extract the task from the label
    task: Task = extract_task_from_label(label)

    # Select the appropriate reply reward calculator based on task type
    if task.task_type == TaskType.RETRIEVAL:
        reply_reward_calculator = calculate_retrieval_reply_reward
    elif task.task_type == TaskType.UPDATE:
        reply_reward_calculator = calculate_update_reply_reward
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")

    # Process the action using the shared base function
    reward, done, next_observation = process_action_base(
        observation=observation,
        action=action,
        python_code=python_code,
        reply=reply,
        thoughts=thoughts,
        task=task,
        thoughts_min_length=THOUGHTS_MIN_LENGTH,
        step_num=step_idx,
        reply_reward_calculator=reply_reward_calculator
    )
        
    step_idx += 1
    reward = torch.tensor(reward)

    # Sampling parameters
    sampling_params = kwargs.get("sampling_params", None)
    if sampling_params is None:
        sampling_params = SamplingParams(stop=["<result>"])
    sampling_params.stop = ["<result>"]

    return {
        "rewards": reward,
        "scores": reward,
        "next_observation": next_observation,
        "done": done,
        "sampling_params": sampling_params,
        "extra_logs": {"formatted_response": format_agent_response(thoughts, python_code, reply, reward)},
    }
    