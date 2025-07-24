import os
import pathlib
import json
from typing import List

from agent.schemas import StaticMemory

from training import MEMORY_PATH

def load_static_memory_from_example_data(memory_dir: pathlib.Path) -> StaticMemory:
    """
    Load a static memory from a memory directory in example_data format.
    """
    base_memory_path = memory_dir / "base_memory.json"
    
    if not base_memory_path.exists():
        raise FileNotFoundError(f"base_memory.json not found in {memory_dir}")
    
    try:
        with open(base_memory_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert mem_id to memory_id to match StaticMemory schema
            if "mem_id" in data:
                data["memory_id"] = data.pop("mem_id")
            return StaticMemory.model_validate(data)
    except Exception as e:
        raise ValueError(f"Error loading static memory from {base_memory_path}: {e}")

def load_all_static_memories(example_data_dir: str = "example_data") -> List[StaticMemory]:
    """
    Load all static memories from example_data directory.
    """
    input_path = pathlib.Path(example_data_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Example data directory not found: {example_data_dir}")
    
    # Find all memory directories
    memory_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("memory_")]
    
    if not memory_dirs:
        raise ValueError(f"No memory directories found in {example_data_dir}")
    
    # Sort for consistent ordering
    memory_dirs.sort(key=lambda x: x.name)
    
    static_memories = []
    for memory_dir in memory_dirs:
        print(f"Loading static memory from {memory_dir.name}...")
        static_memory = load_static_memory_from_example_data(memory_dir)
        static_memories.append(static_memory)
    
    return static_memories

def load_static_memory(path: str) -> StaticMemory:
    """
    Load a single static memory from the old format (for backward compatibility).
    """
    try:
        with open(path, "r") as f:
            return StaticMemory.model_validate_json(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Static memory file not found at {path}")

def instantiate_memory(memory_base_path: str = MEMORY_PATH, example_data_dir: str = "example_data"):
    """
    Instantiate all memory directories from example_data.
    """
    try:
        # Load all static memories from example_data
        static_memories = load_all_static_memories(example_data_dir)
        
        print(f"Found {len(static_memories)} static memories to instantiate")
        
        # Create base memory directory if it doesn't exist
        if not os.path.exists(memory_base_path):
            os.makedirs(memory_base_path, exist_ok=True)
            print(f"Created base memory directory: {memory_base_path}")
        
        # Instantiate each memory
        for static_memory in static_memories:
            memory_path = os.path.join(memory_base_path, static_memory.memory_id)
            
            # Remove existing memory if it exists, then create fresh
            if os.path.exists(memory_path):
                print(f"Resetting existing memory: {static_memory.memory_id}")
                static_memory.reset(memory_base_path)
            else:
                print(f"Creating new memory: {static_memory.memory_id}")
                static_memory.instantiate(memory_base_path)
        
        print(f"\n✓ Successfully instantiated {len(static_memories)} memories in {memory_base_path}")
        
    except Exception as e:
        print(f"Error instantiating memories: {e}")
        raise

def reset_all_memories(memory_base_path: str = MEMORY_PATH, example_data_dir: str = "example_data"):
    """
    Reset all memory directories from example_data.
    """
    try:
        # Load all static memories from example_data
        static_memories = load_all_static_memories(example_data_dir)
        
        print(f"Resetting {len(static_memories)} memories...")
        
        # Reset each memory
        for static_memory in static_memories:
            print(f"Resetting memory: {static_memory.memory_id}")
            static_memory.reset(memory_base_path)
        
        print(f"\n✓ Successfully reset {len(static_memories)} memories")
        
    except Exception as e:
        print(f"Error resetting memories: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup memory from example data")
    parser.add_argument("--example_data_dir", default="example_data", 
                       help="Directory containing memory_* subdirectories")
    parser.add_argument("--memory_path", default=MEMORY_PATH,
                       help="Base path where memories will be instantiated")
    parser.add_argument("--reset", action="store_true",
                       help="Reset existing memories before creating new ones")
    
    args = parser.parse_args()
    
    if args.reset:
        reset_all_memories(args.memory_path, args.example_data_dir)
    else:
        instantiate_memory(args.memory_path, args.example_data_dir)