from pathlib import Path
import os 

TRAINING_DIR = Path(__file__).parent.absolute()
OBSIDIAN_ROOT = TRAINING_DIR.parent

GRONINGEN_STATIC_MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "data", "groningen_memory.json")
BERLIN_STATIC_MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "data", "berlin_memory.json")
MEMORY_PATH = os.path.join(OBSIDIAN_ROOT, "memory")
GRONINGEN_MEMORY_PATH = os.path.join(MEMORY_PATH, "groningen")
BERLIN_MEMORY_PATH = os.path.join(MEMORY_PATH, "berlin")

# Make sure MEMORY_PATH exists
if not os.path.exists(MEMORY_PATH):
    os.makedirs(MEMORY_PATH)