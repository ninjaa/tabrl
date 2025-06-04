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
    
    def start_training(
        self, 
        task_description: str, 
        robot_xml: str, 
        episodes: int = 1000
    ) -> str:
        """Start a new training job"""
        
        training_id = str(uuid.uuid4())[:8]
        
        # Initialize training status
        status = TrainingStatus(
            id=training_id,
            status="running",
            progress=0.0,
            episode=0,
            total_episodes=episodes,
            reward=0.0,
            loss=None,
            eta_seconds=episodes * 3,  # Rough estimate: 3 seconds per episode
            model_path=None,
            error=None,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.active_trainings[training_id] = status
        
        # Start training in background
        asyncio.create_task(self._run_training(training_id, task_description, robot_xml))
        
        return training_id
    
    async def _run_training(self, training_id: str, task_description: str, robot_xml: str):
        """Run the actual training process (mock implementation for now)"""
        
        status = self.active_trainings[training_id]
        
        try:
            # Mock training loop
            for episode in range(status.total_episodes):
                
                # Simulate training step
                await asyncio.sleep(0.1)  # Simulate computation time
                
                # Update progress
                status.episode = episode + 1
                status.progress = episode / status.total_episodes
                status.reward = self._mock_reward_curve(episode, status.total_episodes)
                status.loss = max(0.1, 1.0 - (episode / status.total_episodes) * 0.9)
                status.eta_seconds = int((status.total_episodes - episode) * 0.1)
                status.updated_at = time.time()
                
                # Log progress every 100 episodes
                if episode % 100 == 0:
                    print(f"🏋️ Training {training_id}: Episode {episode}/{status.total_episodes}, Reward: {status.reward:.2f}")
            
            # Training completed - export model
            model_path = await self._export_model(training_id, task_description)
            
            status.status = "completed"
            status.progress = 1.0
            status.model_path = model_path
            status.eta_seconds = 0
            status.updated_at = time.time()
            
            print(f"✅ Training {training_id} completed! Model saved to {model_path}")
            
        except Exception as e:
            status.status = "failed"
            status.error = str(e)
            status.updated_at = time.time()
            print(f"❌ Training {training_id} failed: {e}")
    
    def _mock_reward_curve(self, episode: int, total_episodes: int) -> float:
        """Generate a realistic reward curve for demo purposes"""
        # Exponential learning curve with noise
        progress = episode / total_episodes
        base_reward = 1.0 - 0.8 * (0.95 ** episode)  # Exponential approach to 1.0
        noise = 0.1 * (0.5 - hash(episode) % 100 / 100)  # Some randomness
        return max(0.0, base_reward + noise)
    
    async def _export_model(self, training_id: str, task_description: str) -> str:
        """Export trained model to ONNX format"""
        
        # Clean task description for filename
        safe_name = "".join(c for c in task_description if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
        
        model_filename = f"{training_id}_{safe_name}.onnx"
        model_path = self.models_dir / model_filename
        
        # Mock ONNX export (in real implementation, this would save actual trained model)
        await asyncio.sleep(1)  # Simulate export time
        
        # Create a placeholder ONNX file
        model_path.write_bytes(b"MOCK_ONNX_MODEL_DATA")
        
        # Save metadata
        metadata = {
            "training_id": training_id,
            "task_description": task_description,
            "created_at": time.time(),
            "model_type": "RL_Policy",
            "framework": "TabRL"
        }
        
        metadata_path = model_path.with_suffix('.json')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
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
