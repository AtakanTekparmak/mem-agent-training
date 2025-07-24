from enum import Enum
import re

from pydantic import BaseModel

# Define constants
DELIMITER = "\n\n~/~/~/~/~/~/~/~/~/~/~/~\n\n"

class TaskType(Enum):
    RETRIEVAL = "retrieval"
    UPDATE = "update"
    CLARIFICATION = "clarification"

class Task(BaseModel):
    task_type: TaskType
    mem_id: str
    answer: str

def construct_label(task_type: TaskType, answer: str, mem_id: str) -> str:
    """
    Constructs a label with a given task type, answer, and mem_id.

    Args:
        task_type: The type of task to construct the label for.
        answer: The answer to the task.
        mem_id: The id of the memory to update.

    Returns:
        A string representing the label.
    """
    return f"{task_type.value}{DELIMITER}{mem_id}{DELIMITER}{answer}"

def extract_task_from_label(label: str) -> Task:
    """
    Extracts a Task object from a label.

    Args:
        label: The label to extract the task from.

    Returns:
        A Task object.
    """ 
    task_type, mem_id, answer = label.split(DELIMITER)
    return Task(task_type=TaskType(task_type), mem_id=mem_id, answer=answer)


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
    

def format_agent_response(thoughts: str, python_code: str, reply: str, reward: float) -> str:
    """
    Formats the agent's response to a string.

    Args:
        thoughts: The agent's thoughts.
        python_code: The agent's python code.
        reply: The agent's reply.
        reward: The reward received by the agent.
    
    Returns:
        A string representing the agent's response.
    """
    return f"Agent's response:\n<think>{thoughts}</think>\n<python>{python_code}</python>\n<reply>{reply}</reply>\n<reward>{reward}</reward>"

def extract_python_blocks(observation: str) -> str:
    """
    Extracts all the python blocks from the observation
    and returns them in a single concatenated string.

    Args:
        observation: The input prompt/expression

    Returns:
        A string containing all the python blocks.
    """
    # Find all the python blocks in the observation
    python_blocks = re.findall(r"<python>(.*?)</python>", observation)
     
    # Join them, with the proper <python> and </python> tags before and after
    return "\n".join([f"<python>{block}</python>" for block in python_blocks])

def remove_all_thinks_except_last(observation: str) -> str:
    """
    Removes all the <think> blocks from the observation except the last one,
    but only processes think blocks that appear after "assistant" is seen.

    Args:
        observation: The conversation history as a string.

    Returns:
        A string containing all observation with the <think> blocks except the last one removed.
    """
    # Find the position where "assistant" first appears
    assistant_pos = observation.find("assistant")
    
    if assistant_pos == -1:
        # No "assistant" found, return observation unchanged
        return observation
    
    # Split the observation into before and after "assistant"
    before_assistant = observation[:assistant_pos + len("assistant")]
    after_assistant = observation[assistant_pos + len("assistant"):]
    
    # Find all the <think> blocks in the part after "assistant"
    think_blocks = re.findall(r"<think>(.*?)</think>", after_assistant, re.DOTALL)

    if len(think_blocks) == 0:
        return observation
    elif len(think_blocks) == 1:
        return observation
    else:
        # Remove all think blocks except the last one from the after_assistant part
        result_after = after_assistant
        
        # Remove all but the last think block
        for i in range(len(think_blocks) - 1):
            # Find and remove the first occurrence of this think block
            pattern = r"<think>" + re.escape(think_blocks[i]) + r"</think>"
            result_after = re.sub(pattern, "", result_after, count=1, flags=re.DOTALL)
        
        # Combine the before and processed after parts
        return before_assistant + result_after
        