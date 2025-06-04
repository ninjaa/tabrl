"""
TabRL Backend - Local Python Service
Handles inference, training, and model management
"""

import os
from pathlib import Path
from typing import Optional
import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from inference import InferenceEngine
from training import TrainingEngine

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
    """List available robot scenes"""
    scenes_dir = Path("../scenes")
    if not scenes_dir.exists():
        return {"scenes": []}
    
    scenes = []
    for scene_dir in scenes_dir.iterdir():
        if scene_dir.is_dir():
            xml_file = scene_dir / "robot.xml"
            if xml_file.exists():
                scenes.append({
                    "name": scene_dir.name,
                    "path": str(xml_file),
                    "thumbnail": str(scene_dir / "thumbnail.png") if (scene_dir / "thumbnail.png").exists() else None
                })
    
    return {"scenes": scenes}

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
