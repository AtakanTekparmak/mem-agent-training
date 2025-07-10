import os

from agent.schemas import StaticMemory

from training import STATIC_MEMORY_PATH, MEMORY_PATH

def load_static_memory(path: str = STATIC_MEMORY_PATH) -> StaticMemory:
    """
    Load the static memory from the given path.
    """
    try:
        with open(path, "r") as f:
            return StaticMemory.model_validate_json(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Static memory file not found at {path}")
    
def instantiate_memory(path: str = MEMORY_PATH):
    """
    Instantiate the memory directory.
    """
    static_memory = load_static_memory()
    if not os.path.exists(path):
        static_memory.instantiate(path)

if __name__ == "__main__":
    instantiate_memory()