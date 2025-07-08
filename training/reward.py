from pathlib import Path
import os
import uuid
import json

from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from training import OBSIDIAN_ROOT

load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_PROMPT_PATH = os.path.join(OBSIDIAN_ROOT, "training", "judge_prompt.txt")
GPT_O3 = "o3-2025-04-16"
DEBUG_DIR = os.path.join(OBSIDIAN_ROOT, "debug")
DEBUG_JUDGE_DIR = os.path.join(DEBUG_DIR, "judge")
os.makedirs(DEBUG_JUDGE_DIR, exist_ok=True)

class JudgeResponse(BaseModel):
    question: str
    reply: str
    ground_truth: str
    reasoning: str
    ground_truth_in_reply: bool

def load_judge_prompt(question: str, reply: str, ground_truth: str) -> str:
    """
    Load the judge prompt and replace the placeholders with the reply and ground truth.
    """
    try:
        with open(JUDGE_PROMPT_PATH, "r") as f:
            judge_prompt = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Judge prompt file not found at {JUDGE_PROMPT_PATH}")
    
    judge_prompt = judge_prompt.replace("{{question}}", question)
    judge_prompt = judge_prompt.replace("{{reply}}", reply)
    judge_prompt = judge_prompt.replace("{{ground_truth}}", ground_truth)
    return judge_prompt

def get_model_response(schema: BaseModel, prompt: str, model: str) -> BaseModel:
    """
    Get a structured response from the OpenAI model

    Args:
        schema: The schema of the response
        prompt: The prompt to send to the model
        model: The model to use

    Returns:
        The structured response
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ],
        text_format=schema
    )

    return response.output_parsed  

def get_reward(
        question: str,
        agent_reply: str,
        ground_truth: str,
        debug: bool = False
    ) -> float:
    """
    Get the reward for the given agent reply and ground truth.
    
    Returns:
        float: 1.0 if ground truth is present in reply, 0.0 otherwise
    """
    judge_prompt = load_judge_prompt(question, agent_reply, ground_truth)
    judge_response = get_model_response(
        schema=JudgeResponse,
        prompt=judge_prompt,
        model=GPT_O3
    )

    if debug:
        debug_id = str(uuid.uuid4())
        debug_file = os.path.join(DEBUG_JUDGE_DIR, f"judge_response_{debug_id}.json")
        try:
            with open(debug_file, "w") as f:
                json.dump(judge_response.model_dump(), f)
        except Exception as e:
            print(f"Error saving debug file: {e}")

    return 1.0 if judge_response.ground_truth_in_reply else 0.0