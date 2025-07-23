import torch

from agent.utils import format_results
from agent.engine import execute_sandboxed_code
from training import MEMORY_PATH
from training.reward import get_retrieval_reward
from training.utils import Task, extract_question


def process_retrieval_action(
    observation: str,
    action: str,
    python_code: str,
    reply: str,
    thoughts: str,
    task: Task,
    thoughts_min_length: int
) -> tuple[float, bool, str]:
    """
    Process a retrieval action and return reward, done status, and next observation.
    
    Args:
        observation: The input prompt/expression
        action: The language model's response
        python_code: Extracted python code from action
        reply: Extracted reply from action
        thoughts: Extracted thoughts from action
        task: The task object containing answer
        thoughts_min_length: Minimum length required for thoughts
        
    Returns:
        tuple: (reward, done, next_observation)
    """
    # Check if the action contains a python code, reply, or thoughts block
    python_code_exists = len(python_code.strip()) > 0
    reply_exists = len(reply.strip()) > 0
    thoughts_exists_and_long_enough = len(thoughts.strip()) > thoughts_min_length

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

        next_observation = (
            observation + action +
            format_results(local_vars, error_msg) +
            ("\n<assistant>")
        )

        # format reward 
        reward += 0.1
    elif reply_exists:
        question = extract_question(observation)
        reward += max(0.1, get_retrieval_reward(question, reply, task.answer))
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

    # If reward is higher than 1, set it to 1
    reward = min(reward, 1.0)
    
    return reward, done, next_observation
