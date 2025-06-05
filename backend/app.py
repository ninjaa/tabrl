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
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np

from inference import InferenceEngine
from training import TrainingEngine
from scene_parser import parse_scene_xml, generate_llm_context
import playground_api
from jax_inference import JAXPolicyInference, run_inference_server
from pydantic import BaseModel
import menagerie_server

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

# Mount static directories
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

# Serve temp_menagerie as static files
temp_menagerie_path = Path(__file__).parent.parent / "temp_menagerie"
if temp_menagerie_path.exists():
    app.mount("/menagerie", StaticFiles(directory=str(temp_menagerie_path)), name="menagerie")
    print(f"📁 Serving MuJoCo Menagerie from: {temp_menagerie_path}")

# Mount static files for scenes
scenes_path = Path(__file__).parent.parent / "scenes"
if scenes_path.exists():
    app.mount("/scenes", StaticFiles(directory=str(scenes_path)), name="scenes")

# Initialize services
inference_engine = InferenceEngine()
training_engine = TrainingEngine()
jax_inference_engine = JAXPolicyInference()  # For JAX policy inference

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
        environments = playground_api.get_available_environments()
        return environments
    except Exception as e:
        logger.error(f"Error listing playground environments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/playground/{category}/{env_name}/xml")
async def get_playground_xml_endpoint(category: str, env_name: str):
    """Get the XML content for a specific playground environment"""
    try:
        xml = playground_api.get_environment_xml(category, env_name)
        if xml is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return {"xml": xml}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get XML: {str(e)}")

@app.get("/api/playground/{category}/{env_name}/info")
async def get_playground_info_endpoint(category: str, env_name: str):
    """Get detailed information about a specific playground environment"""
    try:
        info = playground_api.get_environment_info(category, env_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return info
    except Exception as e:
        logger.error(f"Error getting environment info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/playground/{category}/{env_name}/asset/{path:path}")
async def get_playground_asset(category: str, env_name: str, path: str):
    """Get an asset file for a playground environment."""
    import mujoco_playground
    from pathlib import Path
    
    # First check if this is an included XML file
    if path.endswith('.xml'):
        # Try to find it in the mujoco_playground package
        playground_path = Path(mujoco_playground.__file__).parent
        
        # Common paths to check
        possible_paths = [
            playground_path / category / env_name / path,
            playground_path / category / path,
            playground_path / "assets" / path,
            playground_path / path,
        ]
        
        # Also check in temp_menagerie if available
        temp_menagerie = Path(__file__).parent.parent / "temp_menagerie"
        if temp_menagerie.exists():
            # Map environment names to menagerie folders
            env_to_folder = {
                "G1JoystickFlatTerrain": "unitree_g1",
                "G1JoystickRoughTerrain": "unitree_g1",
                "Go1JoystickFlatTerrain": "unitree_go1",
                "Go1JoystickRoughTerrain": "unitree_go1",
                "BarkourJoystick": "google_barkour_vb",
                "SpotJoystickGaitTracking": "boston_dynamics_spot",
                "SpotFlatTerrainJoystick": "boston_dynamics_spot",
                "H1JoystickGaitTracking": "unitree_h1",
                "H1InplaceGaitTracking": "unitree_h1",
                "Op3Joystick": "robotis_op3",
                "BerkeleyHumanoidJoystickFlatTerrain": "berkeley_humanoid",
                "BerkeleyHumanoidJoystickRoughTerrain": "berkeley_humanoid",
                "BerkeleyHumanoidInplaceGaitTracking": "berkeley_humanoid",
            }
            
            folder = env_to_folder.get(env_name)
            if folder:
                possible_paths.extend([
                    temp_menagerie / folder / path,
                    temp_menagerie / folder / "assets" / path,
                ])
        
        for p in possible_paths:
            if p.exists():
                return FileResponse(str(p), media_type="application/xml")
    
    # For other assets (meshes, textures), similar logic
    return JSONResponse(
        status_code=404,
        content={"detail": f"Asset not found: {path}"}
    )

@app.get("/api/playground/{category}/{env_name}/scene-package")
async def get_playground_scene_package(category: str, env_name: str):
    """Get a complete scene package with all assets resolved and included."""
    try:
        # Get the base XML from playground
        xml_content = playground_api.get_playground_xml(category, env_name)
        if not xml_content:
            raise HTTPException(status_code=404, detail=f"Environment {category}/{env_name} not found")
        
        # Get the complete package with all assets
        temp_menagerie = Path(__file__).parent.parent / "temp_menagerie"
        if not temp_menagerie.exists():
            logger.warning(f"temp_menagerie not found at {temp_menagerie}")
            # Return just the XML if we don't have menagerie
            return {"xml": xml_content, "assets": {}}
        
        package = menagerie_server.get_complete_scene_package(
            category, env_name, xml_content, temp_menagerie
        )
        
        # Log the results for debugging
        logger.info(f"Scene package for {category}/{env_name}: {len(package.get('assets', {}))} assets")
        
        return package
        
    except Exception as e:
        logger.error(f"Error getting scene package: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/playground/{category}/{env_name}/complete-xml")
async def get_playground_complete_xml(category: str, env_name: str):
    """Get the complete XML with all includes resolved."""
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import mujoco_playground
    
    try:
        # Get the base XML
        xml_content = playground_api.get_playground_xml(category, env_name)
        if not xml_content:
            raise HTTPException(status_code=404, detail=f"Environment {category}/{env_name} not found")
        
        # Parse the XML
        root = ET.fromstring(xml_content)
        
        # Find the appropriate base path
        temp_menagerie = Path(__file__).parent.parent / "temp_menagerie"
        env_to_folder = {
            "G1JoystickFlatTerrain": "unitree_g1",
            "G1JoystickRoughTerrain": "unitree_g1",
            "Go1JoystickFlatTerrain": "unitree_go1",
            "Go1JoystickRoughTerrain": "unitree_go1",
            "BarkourJoystick": "google_barkour_vb",
            "SpotJoystickGaitTracking": "boston_dynamics_spot",
            "SpotFlatTerrainJoystick": "boston_dynamics_spot",
            "H1JoystickGaitTracking": "unitree_h1",
            "H1InplaceGaitTracking": "unitree_h1",
            "Op3Joystick": "robotis_op3",
            "BerkeleyHumanoidJoystickFlatTerrain": "berkeley_humanoid",
            "BerkeleyHumanoidJoystickRoughTerrain": "berkeley_humanoid",
            "BerkeleyHumanoidInplaceGaitTracking": "berkeley_humanoid",
        }
        
        folder = env_to_folder.get(env_name)
        base_path = temp_menagerie / folder if folder and temp_menagerie.exists() else None
        
        # Process includes recursively
        def process_includes(element, current_path):
            includes = element.findall('.//include')
            for include in includes:
                file_attr = include.get('file')
                if file_attr and base_path:
                    include_path = current_path / file_attr
                    if include_path.exists():
                        # Read the included file
                        include_content = include_path.read_text()
                        include_root = ET.fromstring(include_content)
                        
                        # Get the parent of the include element
                        parent = element
                        for elem in element.iter():
                            if include in elem:
                                parent = elem
                                break
                        
                        # Replace include with contents
                        idx = list(parent).index(include)
                        parent.remove(include)
                        
                        # Add all children from included file
                        for child in include_root:
                            parent.insert(idx, child)
                            idx += 1
                            
                        # Process nested includes
                        process_includes(parent, include_path.parent)
        
        if base_path and base_path.exists():
            process_includes(root, base_path)
        
        # Convert back to string
        complete_xml = ET.tostring(root, encoding='unicode')
        
        return JSONResponse(content={"xml": complete_xml})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy/generate", response_model=PolicyGenerationResponse)
async def generate_policy(request: PolicyGenerationRequest):
    """Generate RL policy and reward functions for the given task"""
    try:
        scene_structure = None
        xml_content = None
        
        # First, check if it's a playground environment
        if "/" not in request.scene_name:
            # It might be a playground environment without category prefix
            # Try to find it in available environments
            environments = playground_api.get_available_environments()
            
            # Search across all categories
            for category, envs in environments.items():
                if request.scene_name in envs:
                    # Found it! Get the XML
                    xml_content = playground_api.get_environment_xml(category, request.scene_name)
                    if xml_content:
                        # Parse the XML content directly
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                            f.write(xml_content)
                            temp_path = f.name
                        
                        scene_structure = parse_scene_xml(temp_path)
                        import os
                        os.unlink(temp_path)
                    break
        
        # If not found as playground env, try local scenes directory
        if scene_structure is None:
            scenes_dir = Path(__file__).parent.parent / "scenes"
            scene_path = scenes_dir / request.scene_name
            
            # Try different XML file names
            xml_files = ["scene.xml", "robot.xml", "ur5e.xml", "scene_left.xml", "scene_right.xml"]
            xml_file = None
            for xml_name in xml_files:
                if (scene_path / xml_name).exists():
                    xml_file = scene_path / xml_name
                    break
            
            if xml_file:
                scene_structure = parse_scene_xml(str(xml_file))
            else:
                raise HTTPException(status_code=404, detail=f"Scene not found: {request.scene_name}")
        
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
        success = jax_inference_engine.load_model(model_path)
        if success:
            model_id = Path(model_path).stem
            info = jax_inference_engine.get_model_info(model_id)
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
        action = jax_inference_engine.predict(model_id, np.array(observation))
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
    """WebSocket endpoint for real-time inference"""
    await websocket.accept()
    
    try:
        while True:
            # Receive observation from client
            data = await websocket.receive_json()
            
            if data.get("type") == "inference_request":
                observation = data.get("observation")
                model_id = data.get("model_id")
                
                # Get action from inference engine
                action = jax_inference_engine.predict(model_id, np.array(observation))
                
                if action is not None:
                    await websocket.send_json({
                        "type": "inference_response",
                        "action": action.tolist()
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No model loaded or inference failed"
                    })
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/api/trained-models")
async def list_trained_models():
    """List all trained JAX models from Modal volume"""
    try:
        from modal import Function
        list_models_fn = Function.from_name("tabrl-playground-training", "list_trained_models")
        models = list_models_fn.remote()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing trained models: {e}")
        # Return empty list if Modal is not connected
        return {"models": []}

@app.get("/api/trained-models/{model_name}/info")
async def get_trained_model_info(model_name: str):
    """Get information about a specific trained model"""
    try:
        from modal import Function
        get_info_fn = Function.from_name("tabrl-playground-training", "get_model_info")
        info = get_info_fn.remote(model_name)
        if info:
            return info
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/models/inference")
async def multi_instance_inference(websocket: WebSocket):
    """WebSocket endpoint for multiple inference instances"""
    await websocket.accept()
    
    # Parse query parameters for scene and instance
    query_params = dict(websocket.query_params)
    scene_name = query_params.get('scene', 'default')
    instance_id = query_params.get('instance', 'default')
    
    logger.info(f"New inference connection: scene={scene_name}, instance={instance_id}")
    
    try:
        # Simple simulation loop for demo
        # In production, this would load the actual trained model for the scene
        step_count = 0
        
        while True:
            # Simulate getting observations and generating actions
            # In a real implementation, this would:
            # 1. Receive observations from the frontend simulation
            # 2. Run inference using the loaded JAX model
            # 3. Send back actions
            
            # For now, generate dummy actions for demo
            action_dim = 6  # Default action dimension
            actions = [np.sin(step_count * 0.1 + i) * 0.5 for i in range(action_dim)]
            
            await websocket.send_json({
                "type": "action",
                "actions": actions,
                "step": step_count,
                "scene": scene_name,
                "instance": instance_id
            })
            
            step_count += 1
            await asyncio.sleep(0.05)  # 20 Hz update rate
            
    except Exception as e:
        logger.error(f"Multi-instance inference error: {e}")
        await websocket.close()

@app.get("/api/playground/models")
async def list_playground_models():
    """List all available playground models with metadata"""
    try:
        from mujoco_playground import registry
        
        models = []
        for category in ['locomotion', 'manipulation', 'dm_control_suite']:
            envs = registry.list_envs(category=category)
            for env_name in envs:
                # Basic metadata - expand as needed
                model_info = {
                    'id': env_name,
                    'name': env_name.replace('_', ' ').title(),
                    'category': category,
                    'dof': 12 if 'quadruped' in env_name.lower() else 6,  # Placeholder
                    'has_base_policy': env_name == 'Go1JoystickFlatTerrain',  # Only Go1 has base for now
                    'thumbnail': f'/static/thumbnails/{env_name}.png' if os.path.exists(f'static/thumbnails/{env_name}.png') else None
                }
                models.append(model_info)
                
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/approaches")
async def generate_training_approaches(request: dict):
    """Generate training approaches using specified LLM provider"""
    model_id = request.get("model_id")
    task_description = request.get("task_description")
    num_approaches = request.get("num_approaches", 3)
    llm_model = request.get("llm_model", "claude-opus-4-20250514")
    
    # For now, return mock data - integrate with actual generation
    approaches = []
    approach_types = ["rhythmic", "smooth", "energetic"]
    
    for i in range(num_approaches):
        approach = {
            "name": f"{approach_types[i].capitalize()} {task_description.split()[0].capitalize()}",
            "description": f"A {approach_types[i]} approach to {task_description}",
            "reward_code": f"""def reward_fn(state, action):
    # {approach_types[i].capitalize()} reward for {task_description}
    base_reward = 1.0
    # Add specific reward logic here
    return base_reward"""
        }
        approaches.append(approach)
    
    return {"approaches": approaches}

@app.post("/api/training/batch")
async def start_batch_training(request: dict):
    """Start multiple training jobs in parallel on Modal"""
    model_id = request.get("model_id")
    approaches = request.get("approaches", [])
    num_steps = request.get("num_steps", 50000)
    use_base_policy = request.get("use_base_policy", False)
    
    # Import Modal function
    try:
        import modal
        train_fn = modal.Function.lookup("tabrl-custom-reward-training", "train_custom_reward")
        
        jobs = []
        for approach in approaches:
            # Spawn Modal job
            job = train_fn.spawn(
                scene_name=model_id,
                reward_function_code=approach["reward_code"],
                reward_function_name="reward_fn",
                num_timesteps=num_steps,
                render_video=True
            )
            
            job_info = {
                "job_id": job.object_id,
                "approach": approach["name"],
                "status": "running",
                "progress": 0
            }
            jobs.append(job_info)
            
            # Store job info for tracking
            training_status[job.object_id] = {
                "task": approach["name"],
                "status": "running",
                "progress": 0,
                "modal_job": job
            }
        
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Failed to start batch training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of a Modal training job"""
    if job_id not in training_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = training_status[job_id]
    modal_job = job_info.get("modal_job")
    
    if modal_job:
        try:
            # Check if job is done
            if modal_job.is_ready():
                result = modal_job.get()
                job_info["status"] = "completed"
                job_info["progress"] = 1.0
                job_info["result"] = result
            else:
                # Estimate progress based on time
                job_info["progress"] = min(0.9, job_info["progress"] + 0.1)
        except Exception as e:
            job_info["status"] = "failed"
            job_info["error"] = str(e)
    
    return job_info

@app.post("/api/training/render")
async def render_training_video(request: dict):
    """Render a video from a trained model"""
    job_id = request.get("job_id")
    model_path = request.get("model_path")
    
    # For now, return a mock video URL
    # In production, this would call render_brax_model.py
    video_url = f"/static/videos/{job_id}.mp4"
    
    # You would actually run:
    # python render_brax_model.py --model_path {model_path} --output {video_path}
    
    return {"video_url": video_url}

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
