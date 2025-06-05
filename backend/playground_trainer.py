#!/usr/bin/env python3
"""
Local client for submitting MuJoCo Playground training jobs to Modal.
"""

import modal
import asyncio
from typing import Dict, Any, Optional
import time

class PlaygroundTrainer:
    """Client for training MuJoCo Playground policies on Modal GPU"""
    
    def __init__(self):
        """Initialize Modal client"""
        try:
            # Connect to the Modal app using correct API
            self.train_fn = modal.Function.from_name("tabrl-playground-training", "train_playground_locomotion")
            self.list_models_fn = modal.Function.from_name("tabrl-playground-training", "list_trained_models")
            self.get_model_info_fn = modal.Function.from_name("tabrl-playground-training", "get_model_info")
            print("✅ Connected to deployed Modal app")
        except Exception as e:
            print(f"❌ Could not connect to Modal app: {e}")
            print("Make sure to deploy first with: modal deploy modal_playground_training.py")
            raise
    
    def train(
        self, 
        env_name: str = "Go1JoystickFlatTerrain",
        category: str = "locomotion",  
        training_steps: int = 30_000_000,
        eval_episodes: int = 5,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a training job to Modal.
        
        Args:
            env_name: Environment name (e.g., 'Go1JoystickFlatTerrain')
            category: Category (locomotion, manipulation, dm_control_suite)
            training_steps: Total training steps
            eval_episodes: Number of evaluation episodes
            run_id: Optional run identifier
            
        Returns:
            Training results
        """
        print(f"🚀 Submitting training job for {category}/{env_name}")
        print(f"   Steps: {training_steps:,}")
        print(f"   Eval episodes: {eval_episodes}")
        
        start_time = time.time()
        
        # Submit to Modal
        result = self.train_fn.remote(
            env_name=env_name,
            category=category,
            training_steps=training_steps, 
            eval_episodes=eval_episodes,
            run_id=run_id
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Training completed in {total_time:.1f} seconds")
        print(f"📊 Final reward: {result['avg_reward']:.2f}")
        print(f"💾 Model saved: {result['model_path']}")
        
        return result
    
    def list_models(self):
        """List all trained models"""
        models = self.list_models_fn.remote()
        print(f"📁 Found {len(models)} trained models:")
        for model in models:
            print(f"   - {model}")
        return models
    
    def get_model_info(self, model_filename: str):
        """Get information about a trained model"""
        info = self.get_model_info_fn.remote(model_filename)
        if 'error' in info:
            print(f"❌ {info['error']}")
            return None
        
        print(f"📋 Model: {model_filename}")
        print(f"   Environment: {info['env_name']}")
        print(f"   Training steps: {info['training_steps']:,}")
        print(f"   Training time: {info['training_time']:.1f}s")
        print(f"   Avg reward: {info['avg_reward']:.2f}")
        print(f"   Obs shape: {info['obs_shape']}")
        print(f"   Action size: {info['action_size']}")
        
        return info

def main():
    """CLI interface for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MuJoCo Playground policies on Modal")
    parser.add_argument("--env", default="Go1JoystickFlatTerrain", 
                       help="Environment name")
    parser.add_argument("--category", default="locomotion",
                       choices=["locomotion", "manipulation", "dm_control_suite"],
                       help="Environment category")
    parser.add_argument("--steps", type=int, default=5_000_000,
                       help="Training steps (default: 5M for testing)")
    parser.add_argument("--eval-episodes", type=int, default=3,
                       help="Evaluation episodes")
    parser.add_argument("--list-models", action="store_true",
                       help="List trained models")
    parser.add_argument("--model-info", 
                       help="Get info about a specific model")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with 100K steps")
    parser.add_argument("--list-envs", action="store_true",
                       help="List available environments by category")
    
    args = parser.parse_args()
    
    if args.list_envs:
        # Show available environments
        try:
            from mujoco_playground import registry
            print("🎮 Available Environments by Category:")
            
            categories = ["locomotion", "manipulation", "dm_control_suite"]
            for cat in categories:
                registry_obj = getattr(registry, cat)
                envs = sorted(registry_obj._envs.keys())
                print(f"\n{cat.upper()} ({len(envs)} environments):")
                for env in envs:
                    print(f"  - {env}")
        except ImportError:
            print("❌ mujoco_playground not installed locally")
        return
    
    trainer = PlaygroundTrainer()
    
    if args.list_models:
        trainer.list_models()
    elif args.model_info:
        trainer.get_model_info(args.model_info)
    elif args.quick_test:
        print("🧪 Running quick test (100K steps)")
        trainer.train(
            env_name=args.env,
            category=args.category,
            training_steps=100_000,
            eval_episodes=2,
            run_id=f"test_{args.category}_{args.env}"
        )
    else:
        trainer.train(
            env_name=args.env,
            category=args.category,
            training_steps=args.steps,
            eval_episodes=args.eval_episodes
        )

if __name__ == "__main__":
    main()
