from training.reward import get_retrieval_reward
from training.utils import Task, extract_question


def calculate_retrieval_python_reward(python_code: str, step_num: int) -> float:
    """Calculate reward for python actions in retrieval tasks."""
    reward_addition = 0.15
    if step_num == 0:
        # check if inside the python code there is one of the following:
        # 1. check_if_file_exists("user.md") or check_if_file_exists('user.md')
        # 2. read_file("user.md") or read_file('user.md')
        # We need the exact string match for these
        desired_strings = [
            "check_if_file_exists('user.md')",
            "check_if_file_exists(\"user.md\")",
            "read_file('user.md')",
            "read_file(\"user.md\")",
        ]

        desired_strings_found = False
        for string in desired_strings:
            if string in python_code:
                desired_strings_found = True
                break

        if desired_strings_found:
            reward_addition += 0.2
    
    return reward_addition


def calculate_retrieval_reply_reward(observation: str, reply: str, task: Task) -> float:
    """Calculate reward for reply actions in retrieval tasks."""
    question = extract_question(observation)
    return max(0.1, get_retrieval_reward(question, reply, task.answer))



