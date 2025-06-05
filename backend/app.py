"""
TabRL Backend - Local Python Service
Handles inference, training, and model management
"""

import os
import time
import asyncio
import json
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Body, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np

from inference import InferenceEngine
from training import TrainingEngine
from scene_parser import parse_scene_xml, generate_llm_context
from playground_api import (
    get_available_environments,
    get_environment_xml,
    get_environment_info
)
from jax_inference import inference_engine, run_inference_server
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Debug flag
DEBUG_LLM_RESPONSES = os.getenv("DEBUG_LLM_RESPONSES", "false").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="TabRL Backend",
    description="Local Python service for robotics training and inference",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for scenes
scenes_path = Path(__file__).parent.parent / "scenes"
if scenes_path.exists():
    app.mount("/scenes", StaticFiles(directory=str(scenes_path)), name="scenes")

# Initialize engines
inference_engine = InferenceEngine()
training_engine = TrainingEngine()

# Request/Response Models
class InferenceRequest(BaseModel):
    prompt: str
    model: Optional[str] = None  # Allow model override per request

class TrainingRequest(BaseModel):
    task_description: str
    scene_name: str  # e.g., "manipulation/universal_robots_ur5e"
    episodes: int = 100
    reward_code: str

class ModelSelectionRequest(BaseModel):
    model: str

class PolicyGenerationRequest(BaseModel):
    prompt: str                    # "Pick up the red block"
    scene_name: str               # "manipulation/universal_robots_ur5e"  
    model: Optional[str] = "claude-3-5-sonnet-20241022"   # LLM model override

class RewardFunction(BaseModel):
    name: str
    type: str  # "dense", "sparse", or "shaped"
    reward: str  # Python code for reward calculation

class PolicyGenerationResponse(BaseModel):
    task: str
    scene: str
    policy_code: str
    reward_functions: List[RewardFunction]
    observation_space: int
    action_space: int

logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "TabRL Backend",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/api/inference")
async def inference_endpoint(request: InferenceRequest):
    """Generate policy code from natural language"""
    try:
        # Generate streaming response
        async def generate_response():
            async for chunk in inference_engine.generate_policy(request.prompt, model=request.model):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/start")
async def start_training(request: TrainingRequest):
    """Start training a new policy"""
    try:
        training_id = training_engine.start_training_with_reward_code(
            task_description=request.task_description,
            scene_name=request.scene_name,
            episodes=request.episodes,
            reward_code=request.reward_code
        )
        
        return {
            "status": "started",
            "training_id": training_id,
            "estimated_duration": f"{request.episodes * 5} seconds"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/status")
async def get_training_status(training_id: str):
    """Get training progress"""
    try:
        status = training_engine.get_status(training_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available trained models"""
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    for model_file in models_dir.glob("*.onnx"):
        models.append({
            "name": model_file.stem,
            "path": str(model_file),
            "size": model_file.stat().st_size,
            "created": model_file.stat().st_mtime
        })
    
    return {"models": models}

@app.get("/api/models/llm")
async def get_available_llm_models():
    """Get available LLM models"""
    return {
        "models": inference_engine.get_available_models(),
        "current": inference_engine.get_current_model()
    }

@app.get("/api/scenes")
async def list_scenes():
    """List available robot scenes organized by category"""
    if not scenes_path.exists():
        return {"scenes": {}}
    
    scenes_by_category = {}
    
    # Iterate through category directories (manipulation, locomotion, simple)
    for category_dir in scenes_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            scenes_in_category = []
            
            # Look for robot/scene XML files in each scene subdirectory
            for scene_dir in category_dir.iterdir():
                if scene_dir.is_dir():
                    # Try different XML file names
                    xml_files = ["scene.xml", "robot.xml", "ur5e.xml", "scene_left.xml", "scene_right.xml"]
                    xml_file = None
                    for xml_name in xml_files:
                        if (scene_dir / xml_name).exists():
                            xml_file = scene_dir / xml_name
                            break
                    
                    if xml_file:
                        scenes_in_category.append({
                            "name": scene_dir.name,
                            "category": category_name,
                            "xml_path": f"scenes/{category_name}/{scene_dir.name}/{xml_file.name}",
                            "thumbnail": f"scenes/{category_name}/{scene_dir.name}/thumbnail.png" 
                            if (scene_dir / "thumbnail.png").exists() else None
                        })
            
            if scenes_in_category:
                scenes_by_category[category_name] = scenes_in_category
    
    return {"scenes": scenes_by_category}

@app.post("/api/model/select")
async def select_model(request: ModelSelectionRequest):
    """Select a model for inference"""
    success = inference_engine.set_model(request.model)
    if success:
        return {"status": "success", "model": request.model}
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

@app.get("/api/current-model")
async def get_current_model():
    """Get currently selected model"""
    return {
        "model": inference_engine.get_current_model(),
        "available_models": inference_engine.get_available_models()
    }

@app.get("/api/playground/environments")
async def list_playground_environments_endpoint():
    """List all available MuJoCo Playground environments organized by category"""
    try:
        environments = get_available_environments()
        return environments
    except Exception as e:
        logger.error(f"Error listing playground environments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/playground/{category}/{env_name}/xml")
async def get_playground_xml_endpoint(category: str, env_name: str):
    """Get the XML content for a specific playground environment"""
    try:
        xml = get_environment_xml(category, env_name)
        if xml is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return {"xml": xml}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get XML: {str(e)}")

@app.get("/api/playground/{category}/{env_name}/info")
async def get_playground_info_endpoint(category: str, env_name: str):
    """Get detailed information about a specific playground environment"""
    try:
        info = get_environment_info(category, env_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return info
    except Exception as e:
        logger.error(f"Error getting environment info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy/generate", response_model=PolicyGenerationResponse)
async def generate_policy(request: PolicyGenerationRequest):
    """Generate RL policy and reward functions for the given task"""
    try:
        # Parse the scene to get structure
        scenes_dir = Path(__file__).parent.parent / "scenes"
        scene_path = scenes_dir / request.scene_name
        
        # Try different XML file names
        xml_files = ["scene.xml", "robot.xml", "ur5e.xml", "scene_left.xml", "scene_right.xml"]
        xml_file = None
        for xml_name in xml_files:
            if (scene_path / xml_name).exists():
                xml_file = scene_path / xml_name
                break
        
        if not xml_file:
            raise HTTPException(status_code=404, detail=f"Scene XML not found for {request.scene_name}")
        
        scene_structure = parse_scene_xml(str(xml_file))
        
        # Create context for LLM
        llm_context = generate_llm_context(scene_structure, request.prompt)
        
        # Use structured JSON generation with schema validation
        policy_data = await inference_engine.generate_structured_policy(
            prompt=request.prompt,
            obs_space=scene_structure.nq + scene_structure.nv,
            action_space=scene_structure.nu,
            scene_context=llm_context,
            model=request.model
        )
        
        # DEBUG: Log raw LLM response to file for inspection
        if DEBUG_LLM_RESPONSES:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"debug_llm_response_{timestamp}.json"
            
            print(f"🔍 DEBUG: Raw LLM response type: {type(policy_data)}")
            print(f"🔍 DEBUG: Raw LLM response keys: {list(policy_data.keys()) if isinstance(policy_data, dict) else 'Not a dict'}")
            
            # Check reward_functions specifically
            reward_funcs = policy_data.get("reward_functions")
            print(f"🔍 DEBUG: reward_functions type: {type(reward_funcs)}")
            print(f"🔍 DEBUG: reward_functions length: {len(reward_funcs) if hasattr(reward_funcs, '__len__') else 'No len'}")
            
            # Save raw response to file
            with open(debug_file, 'w') as f:
                json.dump(policy_data, f, indent=2, default=str)
            print(f"🔍 DEBUG: Raw response saved to {debug_file}")
            
            # If it's a string, show first 200 chars
            if isinstance(reward_funcs, str):
                print(f"🔍 DEBUG: reward_functions preview: {reward_funcs[:200]}...")
            elif isinstance(reward_funcs, list):
                print(f"🔍 DEBUG: reward_functions is already a list with {len(reward_funcs)} items")
            else:
                print(f"🔍 DEBUG: reward_functions is unexpected type: {type(reward_funcs)}")
        
        # Fix reward_functions if it's returned as a string instead of array
        if isinstance(policy_data.get("reward_functions"), str):
            try:
                # First try normal JSON parsing
                policy_data["reward_functions"] = json.loads(policy_data["reward_functions"])
                if DEBUG_LLM_RESPONSES:
                    print("✅ DEBUG: Standard JSON parsing successful")
            except json.JSONDecodeError:
                if DEBUG_LLM_RESPONSES:
                    print("🔧 DEBUG: Standard JSON failed, trying triple-quote fix...")
                try:
                    import re
                    
                    def escape_triple_quoted_content(match):
                        content = match.group(1)
                        # Escape quotes and newlines for JSON
                        escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        return f'"{escaped}"'
                    
                    # Fix """content""" -> "escaped_content"
                    reward_str = policy_data["reward_functions"]
                    pattern = r'"""(.*?)"""'
                    fixed_str = re.sub(pattern, escape_triple_quoted_content, reward_str, flags=re.DOTALL)
                    
                    policy_data["reward_functions"] = json.loads(fixed_str)
                    if DEBUG_LLM_RESPONSES:
                        print("✅ DEBUG: Triple-quote fix successful")
                except Exception as e:
                    if DEBUG_LLM_RESPONSES:
                        print(f"❌ DEBUG: Triple-quote fix failed: {e}")
                    policy_data["reward_functions"] = []
        
        # Ensure reward_functions is always an array
        if not isinstance(policy_data.get("reward_functions"), list):
            policy_data["reward_functions"] = []
        
        return PolicyGenerationResponse(
            task=request.prompt,
            scene=request.scene_name,
            policy_code=policy_data["policy_code"],
            reward_functions=[RewardFunction(**reward_func) for reward_func in policy_data["reward_functions"]],
            observation_space=scene_structure.nq + scene_structure.nv,
            action_space=scene_structure.nu
        )
        
    except Exception as e:
        print(f"Policy generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Policy generation failed: {str(e)}")

@app.post("/api/inference/load_model")
async def load_model_for_inference(model_path: str = Body(..., embed=True)):
    """Load a JAX model for server-side inference"""
    try:
        success = inference_engine.load_model(model_path)
        if success:
            model_id = Path(model_path).stem
            info = inference_engine.get_model_info(model_id)
            return {"status": "success", "model_info": info}
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/predict")
async def run_policy_inference(
    model_id: str = Body(...), 
    observation: list = Body(...)
):
    """Run JAX policy inference on the server"""
    try:
        action = inference_engine.predict(model_id, np.array(observation))
        if action is not None:
            return {
                "status": "success",
                "action": action.tolist()
            }
        else:
            raise HTTPException(status_code=400, detail="Inference failed")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    """WebSocket for real-time policy inference"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            model_id = data.get('model_id')
            observation = data.get('observation')
            
            if not model_id or observation is None:
                await websocket.send_json({
                    'status': 'error',
                    'message': 'Invalid request format'
                })
                continue
            
            action = inference_engine.predict(model_id, np.array(observation))
            
            if action is not None:
                await websocket.send_json({
                    'status': 'success',
                    'action': action.tolist()
                })
            else:
                await websocket.send_json({
                    'status': 'error',
                    'message': 'Model not loaded or inference failed'
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    print("🚀 Starting TabRL Backend...")
    print("📡 Frontend can connect to: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
