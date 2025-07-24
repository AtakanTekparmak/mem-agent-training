import os
from typing import Callable, Tuple

from agent.utils import format_results
from agent.engine import execute_sandboxed_code
from training import MEMORY_PATH
from training.utils import Task


def process_action_base(
    observation: str,
    action: str,
    python_code: str,
    reply: str,
    thoughts: str,
    task: Task,
    thoughts_min_length: int,
    step_num: int,
    python_reward_calculator: Callable[[str, int], float],
    reply_reward_calculator: Callable[[str, str, Task], float]
) -> Tuple[float, bool, str]:
    """
    Base function to process actions for both retrieval and update tasks.
    
    Args:
        observation: The input prompt/expression
        action: The language model's response
        python_code: Extracted python code from action
        reply: Extracted reply from action
        thoughts: Extracted thoughts from action
        task: The task object containing answer
        thoughts_min_length: Minimum length required for thoughts
        step_num: Current step number
        python_reward_calculator: Function to calculate reward for python actions
        reply_reward_calculator: Function to calculate reward for reply actions
        
    Returns:
        tuple: (reward, done, next_observation)
    """
    # Check if the action contains a python code, reply, or thoughts block
    python_code_exists = len(python_code.strip()) > 0
    reply_exists = len(reply.strip()) > 0
    thoughts_exists = len(thoughts.strip()) > 0
    thoughts_long_enough = len(thoughts.strip()) > thoughts_min_length

    # Initialize the reward and done flag
    reward = 0.05 if thoughts_exists else 0.0
    if thoughts_long_enough:
        reward += 0.05

    done = False

    if python_code_exists and reply_exists:
        next_observation = (
            observation + action + 
            "\n<|im_start|>user\n" +
            "\n [ERROR] Choose one action: either explore with <python> ... </python> OR provide final answer with <reply> ... </reply>." +
            "\n [HINT] If you haven't found the answer yet, use <python> to interact with the memory. If you're confident, use <reply>." +
            "\n<|im_start|>assistant\n<think>"
        )
    elif python_code_exists:
        local_vars, error_msg = execute_sandboxed_code(
            code=python_code,
            allowed_path=os.path.join(MEMORY_PATH, task.mem_id),
            import_module="agent.tools",
        )

        next_observation = (
            observation + action +
            "\n<|im_start|>user\n" +
            format_results(local_vars, error_msg) +
            ("\n<|im_start|>assistant\n<think>")
        )

        # Use the provided python reward calculator
        reward += python_reward_calculator(python_code, step_num)
        
    elif reply_exists:
        # Use the provided reply reward calculator
        reward += reply_reward_calculator(observation, reply, task)
        done = True

        next_observation = (
            observation + action + "\n<|im_end|>"
        )
    else:
        # Handle the case where the model output didn't contain a recognised
        # block so that `next_observation` is always defined.
        next_observation = (
            observation + action +
            "\n<|im_start|>user\n" +
            "\n [ERROR] Missing action blocks. You must either:" +
            "\n   1. Use <python>...</python> to interact with the memory" +
            "\n   2. Use <reply>...</reply> to provide your final answer to the user" +
            "\n<|im_start|>assistant\n<think>"
        )

    # If reward is higher than 1, set it to 1
    reward = min(reward, 1.0)
    
    return reward, done, next_observation 