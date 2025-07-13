import os

from agent.schemas import StaticMemory

from training import GRONINGEN_STATIC_MEMORY_PATH, BERLIN_STATIC_MEMORY_PATH, GRONINGEN_MEMORY_PATH, BERLIN_MEMORY_PATH

def load_static_memory(path: str) -> StaticMemory:
    """
    Load the static memory from the given path.
    """
    try:
        with open(path, "r") as f:
            return StaticMemory.model_validate_json(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Static memory file not found at {path}")
    
def instantiate_memory():
    """
    Instantiate the memory directory.
    """
    groningen_static_memory = load_static_memory(GRONINGEN_STATIC_MEMORY_PATH)
    berlin_static_memory = load_static_memory(BERLIN_STATIC_MEMORY_PATH)
    if not os.path.exists(GRONINGEN_MEMORY_PATH):
        groningen_static_memory.instantiate(GRONINGEN_MEMORY_PATH)
    if not os.path.exists(BERLIN_MEMORY_PATH):
        berlin_static_memory.instantiate(BERLIN_MEMORY_PATH)

if __name__ == "__main__":
    instantiate_memory()