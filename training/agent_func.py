from typing import Dict, Any
import os

from agent.utils import extract_reply, extract_python_code, format_results
from agent.engine import execute_sandboxed_code
from agent.schemas import StaticMemory

from training import OBSIDIAN_ROOT
from training.reward import get_reward

import torch

# Global states for the environment
step_idx = 0
max_steps = 10

# Constants
STATIC_MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "data", "base_memory.json")
MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "memory")

def load_static_memory(path: str = STATIC_MEMORY_PATH) -> StaticMemory:
    """
    Load the static memory from the given path.
    """
    try:
        with open(path, "r") as f:
            return StaticMemory.model_validate_json(f.read())
    except FileNotFoundError:
        return StaticMemory(user_md="")
    
def ensure_memory_initialized():
    """
    Ensure memory is initialized once per process to avoid conflicts.
    """
    global _memory_initialized
    if not _memory_initialized:
        static_memory = load_static_memory()
        static_memory.instantiate(MEMORY_PATH)

        _memory_initialized = True

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

    # Ensure the memory directory is set up once at the start of an episode
    ensure_memory_initialized()

    # Extract the python code and reply
    python_code = extract_python_code(action)
    reply = extract_reply(action)
    reward = torch.tensor(0)
    done = False

    if python_code and reply:
        next_observation = (
            observation + action + 
            "\n [ERROR] You cannot provide a <python> and a <reply> block at the same time."
        )
    elif python_code:
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
    elif reply:
        reward = torch.tensor(get_reward(observation, reply, label))
        done = True

        next_observation = (
            observation + action + "\n</s>"
        )
    else:
        # Handle the case where the model output didn't contain a recognised
        # block so that `next_observation` is always defined.
        next_observation = (
            observation + action +
            "\n [ERROR] Missing <python> or <reply> block."
        )

    step_idx += 1

    return {
        "rewards": reward,
        "scores": reward,
        "next_observation": next_observation,
        "done": done,
        "sampling_params": kwargs.get("sampling_params", None),
        "extra_logs": {},
    }
    