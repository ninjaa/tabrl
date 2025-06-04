"""
Training Engine - Handles RL training pipeline and model export
"""

import uuid
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from scene_parser import parse_scene_xml
from rl_training import RealTrainer

@dataclass
class TrainingStatus:
    """Training status tracking"""
    id: str
    status: str  # "running", "completed", "failed"
    progress: float  # 0.0 - 1.0
    episode: int
    total_episodes: int
    reward: float
    loss: Optional[float]
    eta_seconds: Optional[int]
    model_path: Optional[str]
    error: Optional[str]
    created_at: float
    updated_at: float

class TrainingEngine:
    """Handles RL training pipeline with progress tracking"""
    
    def __init__(self):
        self.active_trainings: Dict[str, TrainingStatus] = {}
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.real_trainer = RealTrainer()
    
    def start_training(
        self, 
        task_description: str, 
        scene_name: str,
        episodes: int = 100
    ) -> str:
        """Start a new training job"""
        
        training_id = str(uuid.uuid4())[:8]
        
        # Initialize training status
        status = TrainingStatus(
            id=training_id,
            status="initializing",
            progress=0.0,
            episode=0,
            total_episodes=episodes,
            reward=0.0,
            loss=None,
            eta_seconds=episodes * 5,  # Estimate: 5 seconds per episode for real training
            model_path=None,
            error=None,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.active_trainings[training_id] = status
        
        # Start training in background
        asyncio.create_task(self._run_training(training_id, task_description, scene_name))
        
        return training_id
    
    def start_training_with_reward_code(
        self, 
        task_description: str, 
        scene_name: str,
        reward_code: str,
        episodes: int = 100
    ) -> str:
        """Start a new training job with provided reward function code"""
        
        training_id = str(uuid.uuid4())[:8]
        
        # Initialize training status
        status = TrainingStatus(
            id=training_id,
            status="initializing",
            progress=0.0,
            episode=0,
            total_episodes=episodes,
            reward=0.0,
            loss=None,
            eta_seconds=episodes * 5,  # Estimate: 5 seconds per episode for real training
            model_path=None,
            error=None,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.active_trainings[training_id] = status
        
        # Start training in background with reward code
        asyncio.create_task(self._run_training(training_id, task_description, scene_name, reward_code))
        
        return training_id
    
    async def _run_training(self, training_id: str, task_description: str, scene_name: str, reward_code: str = None):
        """Run the training process"""
        
        try:
            status = self.active_trainings[training_id]
            
            # Update status to parsing scene
            status.status = "parsing_scene"
            status.updated_at = time.time()
            
            # Parse scene to get XML path
            scene_xml_path = f"../scenes/{scene_name}/scene.xml"
            
            # Check for alternative XML files
            scene_dir = Path(f"../scenes/{scene_name}")
            xml_files = ["scene.xml", "robot.xml", "ur5e.xml", "scene_left.xml", "scene_right.xml"]
            for xml_name in xml_files:
                if (scene_dir / xml_name).exists():
                    scene_xml_path = str(scene_dir / xml_name)
                    break
            
            if not Path(scene_xml_path).exists():
                raise Exception(f"Scene XML not found: {scene_xml_path}")
            
            print(f"🎯 Using scene: {scene_xml_path}")
            
            # Update status to training
            status.status = "training"
            status.updated_at = time.time()
            
            async def progress_callback(episode: int, total_episodes: int, 
                                      episode_reward: float, avg_reward: float, loss: float = 0.0):
                """Update training progress"""
                status.episode = episode + 1
                status.progress = episode / total_episodes
                status.reward = avg_reward
                status.loss = loss
                status.eta_seconds = int((total_episodes - episode) * 5)  # 5 sec per episode
                status.updated_at = time.time()
            
            # Train the policy
            training_results = await self.real_trainer.train_policy(
                scene_xml_path=scene_xml_path,
                reward_function_code=reward_code,
                training_id=training_id,
                task_description=task_description,
                episodes=status.total_episodes,
                progress_callback=progress_callback
            )
            
            if not training_results.get("success", False):
                raise Exception(f"Training failed: {training_results.get('error', 'Unknown error')}")
            
            # Update status to exporting model
            status.status = "exporting_model"
            status.updated_at = time.time()
            
            # Export model and finalize
            model_path = await self._export_model(training_id, task_description, training_results)
            
            status.status = "completed"
            status.progress = 1.0
            status.model_path = model_path
            status.reward = training_results.get("final_avg_reward", 0.0)
            status.eta_seconds = 0
            status.updated_at = time.time()
            
            print(f"✅ Training {training_id} completed! Final reward: {status.reward:.2f}")
            print(f"📁 Model saved to: {model_path}")
            if training_results.get("video_path"):
                print(f"🎬 Training video: {training_results['video_path']}")
        
        except Exception as e:
            status.status = "failed"
            status.error = str(e)
            status.updated_at = time.time()
            print(f"❌ Training {training_id} failed: {e}")
    
    async def _export_model(self, training_id: str, task_description: str, 
                           training_results: Dict) -> str:
        """Export trained model and metadata"""
        
        # Clean task description for filename
        safe_name = "".join(c for c in task_description if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
        
        model_filename = f"{training_id}_{safe_name}_policy.json"
        model_path = self.models_dir / model_filename
        
        # Save model metadata and training results
        model_data = {
            "training_id": training_id,
            "task_description": task_description,
            "created_at": time.time(),
            "model_type": "RL_Policy",
            "framework": "TabRL",
            "training_results": training_results,
            "final_reward": training_results.get("final_avg_reward", 0.0),
            "total_episodes": training_results.get("total_episodes", 0),
            "model_info": training_results.get("model_info", {}),
            "video_path": training_results.get("video_path")
        }
        
        model_path.write_text(json.dumps(model_data, indent=2))
        return str(model_path)
    
    def get_status(self, training_id: str) -> Dict[str, Any]:
        """Get current training status"""
        if training_id not in self.active_trainings:
            return {"error": "Training ID not found"}
        
        status = self.active_trainings[training_id]
        return asdict(status)
    
    def stop_training(self, training_id: str) -> bool:
        """Stop a running training job"""
        if training_id not in self.active_trainings:
            return False
        
        status = self.active_trainings[training_id]
        if status.status == "running":
            status.status = "stopped"
            status.updated_at = time.time()
            return True
        
        return False
    
    def list_trainings(self) -> Dict[str, Dict[str, Any]]:
        """List all training jobs"""
        return {tid: asdict(status) for tid, status in self.active_trainings.items()}
    
    def cleanup_old_trainings(self, max_age_hours: int = 24):
        """Remove old training records"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        to_remove = []
        for training_id, status in self.active_trainings.items():
            if status.updated_at < cutoff_time and status.status in ["completed", "failed", "stopped"]:
                to_remove.append(training_id)
        
        for training_id in to_remove:
            del self.active_trainings[training_id]
