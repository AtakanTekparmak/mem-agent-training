import random
from typing import Any, Dict
import json

import torch
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase
from vllm import SamplingParams

from agent.utils import extract_reply, extract_python_code, extract_thoughts
from training.action_processor import process_action_base
from training.retrieval import calculate_retrieval_reply_reward
from training.update import calculate_update_reply_reward
from training.utils import Task, TaskType, extract_task_from_label, format_agent_response, remove_all_thinks_except_last

# Load hyperparameters
try:
    with open("config.json", "r") as f:
        config = json.load(f)
        THOUGHTS_MIN_LENGTH = config["hyperparameters"]["thoughts_min_length"]
except:
    raise ValueError("config.json not found or the thoughts_min_length key is not present in hyperparameters")


class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = 10

    async def reset(self, states: dict, **kwargs):
        """Initialize the environment and return initial observation

        Args:
            states: Dictionary containing observation and label

        Returns:
            dict: Initial state with observation
        """
        # Reset step counter for new episode
        self.step_idx = 0
        return {"observation": states["observation"]}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Execute one step of the agent interaction

        Args:
            states: Dictionary containing observation_text, action_text, and label

        Returns:
            Dict[str, Any]: A dictionary containing:
                - rewards: Reward value for advantage calculation
                - scores: Reward value for dynamic filtering
                - environment_feedback: The environment feedback text
                - done: Boolean indicating if the episode is complete
                - sampling_params: Parameters for vLLM sampling
                - extra_logs: Additional logging information
        """
        print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

        if self.step_idx >= self.max_steps:
            done = True
            environment_feedback = (
                "\n [WARNING] You have reached the maximum number of steps."
            )
            return {
                "rewards": torch.tensor(0.0),
                "scores": torch.tensor(0.0),
                "environment_feedback": environment_feedback,
                "done": done,
                "sampling_params": kwargs.get("sampling_params", None),
                "extra_logs": {},
            }

        observation_text = states["observation_text"]
        action_text = states["action_text"]
        label = states["label"]

        # Remove all the <think> blocks except the last one
        observation = remove_all_thinks_except_last(observation_text)
        
        # Truncate the action after the closing tags
        # This preserves all action blocks (including empty ones) by finding the last closing tag
        action = action_text
        
        # Find the positions of both closing tags
        python_end_pos = -1
        reply_end_pos = -1
        
        if "</python>" in action:
            python_end_pos = action.find("</python>") + len("</python>")
            
        if "</reply>" in action:
            reply_end_pos = action.find("</reply>") + len("</reply>")
        
        # Truncate at the position of whichever tag appears last
        # This ensures both blocks are preserved, even if one is empty
        if python_end_pos > 0 or reply_end_pos > 0:
            truncate_pos = max(python_end_pos, reply_end_pos)
            action = action[:truncate_pos]

        # Extract the python code and reply
        python_code = extract_python_code(action)
        reply = extract_reply(action)
        thoughts = extract_thoughts(action)

        # Extract the task from the label
        task: Task = extract_task_from_label(label)

        # Select the appropriate reply reward calculator based on task type
        if task.task_type == TaskType.RETRIEVAL:
            reply_reward_calculator = calculate_retrieval_reply_reward
        elif task.task_type == TaskType.UPDATE:
            reply_reward_calculator = calculate_update_reply_reward
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

        # Process the action using the shared base function
        reward, done, next_observation = process_action_base(
            observation=observation,
            action=action,
            python_code=python_code,
            reply=reply,
            thoughts=thoughts,
            task=task,
            thoughts_min_length=THOUGHTS_MIN_LENGTH,
            step_num=self.step_idx,
            reply_reward_calculator=reply_reward_calculator
        )
            
        self.step_idx += 1
        reward = torch.tensor(reward, dtype=torch.float32)

        # Environment feedback is the difference between next_observation and current observation+action
        environment_feedback = next_observation[len(observation + action):]

        # Sampling parameters
        sampling_params = kwargs.get("sampling_params", None)
        if sampling_params is None:
            sampling_params = SamplingParams(stop=["<result>"])
        sampling_params.stop = ["<result>"]

        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": environment_feedback,
            "done": done,
            "sampling_params": sampling_params,
            "extra_logs": {},
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

    async def execute(self, prompt, label, sampling_params):
        # You can override the execute function to add custom agent running logic
        return await super().execute(prompt, label, sampling_params)
    