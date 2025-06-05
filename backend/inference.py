"""
Inference Engine - Handles LLM-based policy generation using LiteLLM and ONNX model inference
"""

import os
import asyncio
from typing import AsyncGenerator, Optional
from pathlib import Path

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class InferenceEngine:
    """Handles both LLM inference for policy generation and ONNX model inference"""
    
    # Available models with their providers
    AVAILABLE_MODELS = {
        "claude-sonnet-4-20250514": {"provider": "anthropic", "name": "Claude 4 Sonnet"},
        "claude-opus-4-20250514": {"provider": "anthropic", "name": "Claude 4 Opus"},
        "claude-3-5-sonnet-20241022": {"provider": "anthropic", "name": "Claude 3.5 Sonnet"},
        "gpt-4o": {"provider": "openai", "name": "GPT-4o"},
        "gemini/gemini-1.5-pro": {"provider": "google", "name": "Gemini 1.5 Pro"},
        "deepseek/deepseek-chat": {"provider": "deepseek", "name": "DeepSeek Chat"},
    }
    
    def __init__(self):
        self.current_model = "claude-sonnet-4-20250514"  # Default to Claude 4 Sonnet
        self.onnx_sessions = {}  # Cache for loaded ONNX models
        
        # Load environment variables
        self._load_env_config()
        
        # Configure LiteLLM
        if LITELLM_AVAILABLE:
            litellm.set_verbose = False  # Disable verbose logging
    
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
            self.current_model = model_name
            return True
        return False
    
    def get_available_models(self) -> dict:
        """Get list of available models"""
        return self.AVAILABLE_MODELS
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.current_model
    
    async def generate_policy(self, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate policy code from natural language using streaming LLM"""
        
        if not LITELLM_AVAILABLE:
            yield "❌ LiteLLM not available. Please install litellm."
            return
        
        # Use specified model or current default
        model_to_use = model if model and model in self.AVAILABLE_MODELS else self.current_model
        
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

        try:
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
                    
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg.lower():
                provider = self.AVAILABLE_MODELS.get(model_to_use, {}).get("provider", "unknown")
                yield f"❌ API key error for {provider}. Please check your .env file for the required API key."
            else:
                yield f"❌ Error generating policy with {model_to_use}: {error_msg}"
    
    async def generate_structured_policy(self, prompt: str, obs_space: int, action_space: int, scene_context: str, model: Optional[str] = None) -> dict:
        """Generate structured policy response with JSON schema validation"""
        
        if not LITELLM_AVAILABLE:
            raise Exception("LiteLLM not available. Please install litellm.")
        
        # Use specified model or current default
        model_to_use = model if model and model in self.AVAILABLE_MODELS else self.current_model
        
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
                        "required": ["name", "type", "reward"]
                    }
                },
                "observation_space": {"type": "integer"},
                "action_space": {"type": "integer"}
            },
            "required": ["task", "scene", "policy_code", "reward_functions", "observation_space", "action_space"]
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

        try:
            # Create messages for LiteLLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {prompt}"}
            ]
            
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
                max_tokens=4096,
                temperature=0.1
            )
            
            # Parse the structured response
            import json
            policy_data = json.loads(response.choices[0].message.content)
            
            # Ensure correct dimensions
            policy_data["observation_space"] = obs_space
            policy_data["action_space"] = action_space
            
            return policy_data
                    
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg.lower():
                provider = self.AVAILABLE_MODELS.get(model_to_use, {}).get("provider", "unknown")
                raise Exception(f"API key error for {provider}. Please check your .env file for the required API key.")
            else:
                raise Exception(f"Error generating structured policy with {model_to_use}: {error_msg}")
    
    def load_onnx_model(self, model_path: str) -> bool:
        """Load an ONNX model for inference"""
        if not ONNX_AVAILABLE:
            return False
            
        try:
            if model_path not in self.onnx_sessions:
                self.onnx_sessions[model_path] = ort.InferenceSession(model_path)
            return True
        except Exception as e:
            print(f"❌ Failed to load ONNX model {model_path}: {e}")
            return False
    
    def run_onnx_inference(self, model_path: str, observations: np.ndarray) -> Optional[np.ndarray]:
        """Run inference on loaded ONNX model"""
        if not ONNX_AVAILABLE or model_path not in self.onnx_sessions:
            return None
            
        try:
            session = self.onnx_sessions[model_path]
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            result = session.run([output_name], {input_name: observations})
            return result[0]
            
        except Exception as e:
            print(f"❌ ONNX inference error: {e}")
            return None
    
    def get_available_onnx_models(self) -> list:
        """Get list of available ONNX models"""
        models_dir = Path("models")
        if not models_dir.exists():
            return []
            
        return [str(p) for p in models_dir.glob("*.onnx")]
