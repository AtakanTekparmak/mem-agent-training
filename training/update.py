import os

from agent.utils import format_results
from agent.engine import execute_sandboxed_code
from training import MEMORY_PATH
from training.utils import Task, extract_python_blocks
from training.reward import get_update_reward


def process_update_action(
    observation: str,
    action: str,
    python_code: str,
    reply: str,
    thoughts: str,
    task: Task,
    thoughts_min_length: int,
    step_num: int
) -> tuple[float, bool, str]:
    """
    Process a update action and return reward, done status, and next observation.
    
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
            "\n [ERROR] Choose one action: either explore with <python> ... </python> OR provide final answer with <reply> ... </reply>." +
            "\n [HINT] If you haven't found the answer yet, use <python> to interact with the memory. If you're confident, use <reply>." +
            "\n<assistant>"
        )
    elif python_code_exists:
        local_vars, error_msg = execute_sandboxed_code(
            code=python_code,
            allowed_path=os.path.join(MEMORY_PATH, task.mem_id),
            import_module="agent.tools",
        )

        next_observation = (
            observation + action +
            format_results(local_vars, error_msg) +
            ("\n<assistant>")
        )

        reward += 0.15
    elif reply_exists:
        python_blocks = extract_python_blocks(observation)
        diff = task.answer
        reward_addition = get_update_reward(
            python_blocks=python_blocks,
            diff=diff,
            debug=True
        )

        done = True

        reward += max(0.1, reward_addition)

        next_observation = (
            observation + action + "\n</s>"
        )
    else:
        # Handle the case where the model output didn't contain a recognised
        # block so that `next_observation` is always defined.
        next_observation = (
            observation + action +
            "\n [ERROR] Missing action blocks. You must either:" +
            "\n   1. Use <python>...</python> to interact with the memory" +
            "\n   2. Use <reply>...</reply> to provide your final answer to the user" +
            "\n<assistant>"
        )

    # If reward is higher than 1, set it to 1
    reward = min(reward, 1.0)

    return reward, done, next_observation