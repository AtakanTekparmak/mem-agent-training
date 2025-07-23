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
    pass