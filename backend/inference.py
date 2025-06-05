"""
Inference Engine - Handles LLM-based policy generation using LiteLLM
"""

import os
import asyncio
from typing import AsyncGenerator, Optional
from pathlib import Path
import numpy as np
import litellm

class InferenceEngine:
    """Handles LLM inference for policy generation"""
    
    # Available models with their providers
    AVAILABLE_MODELS = {
        "claude-sonnet-4-20250514": {"provider": "anthropic", "name": "Claude 4 Sonnet"},
        "claude-opus-4-20250514": {"provider": "anthropic", "name": "Claude 4 Opus"},
        "o3-mini": {"provider": "openai", "name": "o3 mini"},
        "o3": {"provider": "openai", "name": "o3"},
        "gemini/gemini-2.5-pro-preview-06-05": {"provider": "google", "name": "Gemini 1.5 Pro"},
    }
    
    def __init__(self):
        self.model = "claude-sonnet-4-20250514"  # Default to Claude 4 Sonnet
        self.api_key = os.getenv("LLM_API_KEY")
        
        # Enable debug mode for development
        litellm.set_verbose = True
    
    def _load_env_config(self):
        """Load API keys from environment"""
        # LiteLLM automatically picks up these environment variables:
        # ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
        # Also set GOOGLE_API_KEY as an alias for Gemini compatibility
        if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
    
    def set_model(self, model_name: str) -> bool:
        """Set the current model for inference"""
        if model_name in self.AVAILABLE_MODELS:
            self.model = model_name
            return True
        return False
    
    def get_available_models(self) -> dict:
        """Get list of available models"""
        return self.AVAILABLE_MODELS
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.model
    
    async def generate_policy(self, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate policy code from natural language using streaming LLM"""
        
        # Use specified model or current default
        model_to_use = model if model and model in self.AVAILABLE_MODELS else self.model
        
        # System prompt for robotics policy generation
        system_prompt = """You are an expert robotics engineer specializing in robot control and reinforcement learning. Generate Python code for robot control policies based on natural language descriptions.

Your code should:
1. Use clear, readable functions with proper docstrings
2. Include comprehensive error handling and input validation
3. Work with MuJoCo physics simulation framework
4. Follow robotics best practices for safety and reliability
5. Be production-ready with proper logging
6. Include comments explaining the control logic and approach

Format your response as valid Python code with detailed explanations of the approach, expected inputs/outputs, and any assumptions made."""

        # Create messages for LiteLLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Generate streaming response using LiteLLM
        response = await litellm.acompletion(
            model=model_to_use,
            messages=messages,
            stream=True,
            max_tokens=2048,
            temperature=0.1,  # Lower temperature for more consistent code generation
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    async def generate_structured_policy(self, prompt: str, obs_space: int, action_space: int, scene_context: str, model: Optional[str] = None) -> dict:
        """Generate structured policy response with JSON schema validation"""
        
        # Use specified model or current default
        model_to_use = model if model and model in self.AVAILABLE_MODELS else self.model
        
        # Define the JSON schema for policy response
        policy_schema = {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "scene": {"type": "string"},
                "policy_code": {"type": "string"},
                "reward_functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["dense", "sparse", "shaped"]},
                            "reward": {"type": "string"}
                        },
                        "required": ["name", "type", "reward"],
                        "additionalProperties": False
                    }
                },
                "observation_space": {"type": "integer"},
                "action_space": {"type": "integer"}
            },
            "required": ["task", "scene", "policy_code", "reward_functions", "observation_space", "action_space"],
            "additionalProperties": False
        }
        
        # System prompt for structured policy generation
        system_prompt = f"""You are an expert robotics engineer. Generate a comprehensive policy for the given robotics task using Brax/JAX.

Scene context:
{scene_context}

Create:
1. A complete policy_code implementation (PolicyNetwork class with forward method)
2. Multiple diverse reward_functions (dense, sparse, shaped types) compatible with Brax

CRITICAL: Each reward function MUST follow the Brax reward function signature:
```python
def reward_fn(state, action):
    # state is a Brax PipelineState object with:
    # - state.q: joint positions (array)
    # - state.qd: joint velocities (array)
    # - state.x.pos: body positions (array of 3D positions)
    # - state.x.rot: body rotations (array of quaternions)
    # - state.xd.vel: body linear velocities (array of 3D velocities)
    # - state.xd.ang: body angular velocities (array of 3D velocities)
    
    # Example accessing root body (index 0) forward velocity:
    # forward_velocity = state.xd.vel[0, 0]  # x-velocity
    
    # Example accessing root body height:
    # height = state.x.pos[0, 2]  # z-position
    
    # Return a float reward value
    return float(reward_value)
```

Common body indices (may vary by robot):
- 0: root/torso/base
- 1-4: legs/arms attachments
- Check scene XML for exact body names and ordering

Common reward patterns:
- Forward progress: state.xd.vel[0, 0] (x-velocity of root)
- Upright bonus: state.x.pos[0, 2] (height) * (1 - abs(state.x.rot[0, 1])) (penalize tilt)
- Energy penalty: -0.001 * jnp.sum(action**2)
- Contact forces: Use state.contact if available

Make the reward functions realistic and task-specific. Each reward function should be self-contained and properly formatted Python code that works with Brax state objects."""

        # Create messages for LiteLLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {prompt}"}
        ]
        
        # O3 models only support temperature=1.0
        temperature = 1.0 if model_to_use in ["o3", "o3-mini"] else 0.1
        
        # Generate structured response using JSON schema
        response = await litellm.acompletion(
            model=model_to_use,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "policy_response",
                    "schema": policy_schema,
                    "strict": True
                }
            },
            max_tokens=8192,
            temperature=temperature
        )
        
        # Parse the structured response
        import json
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response")
        policy_data = json.loads(content)
        
        # Ensure correct dimensions
        policy_data["observation_space"] = obs_space
        policy_data["action_space"] = action_space
        
        return policy_data
                
    def run_onnx_inference(self, model_path: str, observations: np.ndarray) -> Optional[np.ndarray]:
        """Deprecated - we use JAX inference now"""
        raise NotImplementedError("ONNX inference has been removed in favor of JAX server-side inference")
    
    def get_available_onnx_models(self) -> list:
        """Deprecated - we use JAX models now"""
        return []
