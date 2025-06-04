"""
Inference Engine - Handles LLM-based policy generation and ONNX model inference
"""

import os
import asyncio
import json
from typing import AsyncGenerator, Optional
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class InferenceEngine:
    """Handles both LLM inference for policy generation and ONNX model inference"""
    
    def __init__(self):
        self.anthropic_client = None
        self.openai_client = None
        self.onnx_sessions = {}  # Cache for loaded ONNX models
        
        # Initialize clients if API keys are available
        self._init_clients()
    
    def _init_clients(self):
        """Initialize LLM clients based on available API keys"""
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
    
    async def generate_policy(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate policy code from natural language using streaming LLM"""
        
        # Refresh clients in case keys were updated
        self._init_clients()
        
        # System prompt for robotics policy generation
        system_prompt = """You are an expert robotics engineer. Generate Python code for robot control policies based on natural language descriptions.

Your code should:
1. Use clear, readable functions
2. Include proper error handling
3. Work with MuJoCo physics simulation
4. Be production-ready

Format your response as valid Python code with comments explaining the approach."""

        try:
            if self.anthropic_client:
                async for chunk in self._generate_with_anthropic(system_prompt, prompt):
                    yield chunk
            elif self.openai_client:
                async for chunk in self._generate_with_openai(system_prompt, prompt):
                    yield chunk
            else:
                yield "❌ No LLM API keys configured. Please set Anthropic or OpenAI API key."
                
        except Exception as e:
            yield f"❌ Error generating policy: {str(e)}"
    
    async def _generate_with_anthropic(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """Generate using Anthropic Claude with streaming"""
        try:
            async with self.anthropic_client.messages.stream(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            yield f"❌ Anthropic API error: {str(e)}"
    
    async def _generate_with_openai(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """Generate using OpenAI GPT with streaming"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                max_tokens=2048
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"❌ OpenAI API error: {str(e)}"
    
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
    
    def get_available_models(self) -> list:
        """Get list of available ONNX models"""
        models_dir = Path("models")
        if not models_dir.exists():
            return []
            
        return [str(p) for p in models_dir.glob("*.onnx")]
