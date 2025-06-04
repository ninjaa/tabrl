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
        "claude-3-5-sonnet-20241022": {"provider": "anthropic", "name": "Claude 3.5 Sonnet"},
        "gpt-4o": {"provider": "openai", "name": "GPT-4o"},
        "gemini/gemini-1.5-pro": {"provider": "google", "name": "Gemini 1.5 Pro"},
        "deepseek/deepseek-chat": {"provider": "deepseek", "name": "DeepSeek Chat"},
    }
    
    def __init__(self):
        self.current_model = "claude-3-5-sonnet-20241022"  # Default to Claude
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
