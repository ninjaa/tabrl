"""
TabRL Backend - Local Python Service
Handles inference, training, and model management
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from inference import InferenceEngine
from training import TrainingEngine
from scene_parser import parse_scene_xml, generate_llm_context

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
    robot_xml: str
    episodes: int = 1000

class ModelSelectionRequest(BaseModel):
    model: str

class PolicyGenerationRequest(BaseModel):
    prompt: str                    # "Pick up the red block"
    scene_name: str               # "manipulation/universal_robots_ur5e"  
    model: Optional[str] = None   # LLM model override

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
        training_id = training_engine.start_training(
            task_description=request.task_description,
            robot_xml=request.robot_xml,
            episodes=request.episodes
        )
        
        return {
            "status": "started",
            "training_id": training_id,
            "estimated_duration": "3-5 minutes"
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
                    xml_files = ["scene.xml", "robot.xml", "ur5e.xml"]
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

@app.get("/api/model/current")
async def get_current_model():
    """Get currently selected model"""
    return {
        "model": inference_engine.get_current_model(),
        "info": inference_engine.get_available_models().get(inference_engine.get_current_model(), {})
    }

@app.post("/api/policy/generate")
async def generate_policy(request: PolicyGenerationRequest):
    """Generate policy with multiple reward approaches using scene context"""
    try:
        # Parse scene XML to get structure
        scenes_dir = Path(__file__).parent.parent / "scenes"
        scene_path = scenes_dir / request.scene_name
        
        # Try different XML file names
        xml_files = ["scene.xml", "robot.xml", "ur5e.xml"]
        xml_file = None
        for xml_name in xml_files:
            if (scene_path / xml_name).exists():
                xml_file = scene_path / xml_name
                break
        
        if not xml_file:
            raise HTTPException(status_code=404, detail=f"Scene XML not found for {request.scene_name}")
        
        # Parse scene structure
        scene_structure = parse_scene_xml(str(xml_file))
        
        # Generate LLM context with scene information
        llm_context = generate_llm_context(scene_structure, request.prompt)
        
        # Create enhanced prompt for multi-reward generation
        enhanced_prompt = f"""
{llm_context}

TASK: {request.prompt}

Generate a complete Python policy package with 3 different reward approaches:

1. **Dense Reward**: Continuous feedback for gradual learning
2. **Sparse Reward**: Large rewards only upon task completion  
3. **Shaped Reward**: Curriculum-style with intermediate milestones

Return this JSON structure:
{{
    "task": "{request.prompt}",
    "scene": "{scene_structure.name}",
    "policy_code": "# Main policy neural network code",
    "reward_functions": [
        {{
            "name": "dense_reward",
            "description": "Continuous distance-based feedback",
            "code": "def compute_reward(state, action): ..."
        }},
        {{
            "name": "sparse_reward", 
            "description": "Success/failure only",
            "code": "def compute_reward(state, action): ..."
        }},
        {{
            "name": "shaped_reward",
            "description": "Milestone-based progression", 
            "code": "def compute_reward(state, action): ..."
        }}
    ],
    "observation_space": {scene_structure.nq + scene_structure.nv},
    "action_space": {scene_structure.nu}
}}
"""
        
        # Generate policy using LLM
        response_text = ""
        async for chunk in inference_engine.generate_policy(enhanced_prompt, model=request.model):
            response_text += chunk
        
        # Try to parse as JSON, fallback to text response
        try:
            import json
            policy_data = json.loads(response_text)
            return policy_data
        except:
            # Fallback if not valid JSON
            return {
                "task": request.prompt,
                "scene": scene_structure.name,
                "raw_response": response_text,
                "scene_structure": {
                    "joints": scene_structure.joints,
                    "bodies": scene_structure.bodies,
                    "sites": scene_structure.sites,
                    "sensors": scene_structure.sensors,
                    "action_dim": scene_structure.nu,
                    "state_dim": scene_structure.nq + scene_structure.nv
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy generation failed: {str(e)}")

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
