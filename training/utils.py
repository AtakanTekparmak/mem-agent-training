from enum import Enum

from pydantic import BaseModel

# Define constants
DELIMITER = "~/~"

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