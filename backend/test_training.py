#!/usr/bin/env python3
"""
Test End-to-End Training Pipeline
"""

import asyncio
import requests
import json
import time
from pathlib import Path

async def test_training_api():
    """Test the complete training pipeline via API"""
    
    print("🧪 Testing End-to-End Training Pipeline...")
    
    # Test cases for different robot types
    test_cases = [
        {
            "task_description": "Pick up a red block and place it on a blue target",
            "scene_name": "manipulation/universal_robots_ur5e",
            "episodes": 20  # Small number for testing
        },
        {
            "task_description": "Walk forward steadily without falling",
            "scene_name": "locomotion/anybotics_anymal_c", 
            "episodes": 20
        }
    ]
    
    base_url = "http://localhost:8000"
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test Case {i+1}: {test_case['scene_name']}")
        print(f"    Task: {test_case['task_description']}")
        
        try:
            # Start training
            print("🚀 Starting training...")
            response = requests.post(f"{base_url}/api/training/start", json=test_case, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ Failed to start training: {response.status_code} - {response.text}")
                continue
            
            result = response.json()
            training_id = result["training_id"]
            print(f"✅ Training started with ID: {training_id}")
            print(f"⏱️ Estimated duration: {result.get('estimated_duration', 'unknown')}")
            
            # Monitor training progress
            print("📊 Monitoring training progress...")
            start_time = time.time()
            last_episode = -1
            
            while True:
                # Check status
                status_response = requests.get(f"{base_url}/api/training/{training_id}/status", timeout=10)
                
                if status_response.status_code != 200:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    break
                
                status = status_response.json()
                current_episode = status.get("episode", 0)
                total_episodes = status.get("total_episodes", 0)
                progress = status.get("progress", 0.0)
                reward = status.get("reward", 0.0)
                training_status = status.get("status", "unknown")
                
                # Log progress updates
                if current_episode != last_episode:
                    elapsed = time.time() - start_time
                    print(f"    Episode {current_episode}/{total_episodes} "
                          f"({progress:.1%}) - Reward: {reward:.2f} - {elapsed:.1f}s elapsed")
                    last_episode = current_episode
                
                # Check if completed
                if training_status == "completed":
                    print(f"✅ Training completed!")
                    print(f"    Final reward: {reward:.2f}")
                    if status.get("model_path"):
                        print(f"    Model saved: {status['model_path']}")
                    break
                elif training_status == "failed":
                    print(f"❌ Training failed: {status.get('error', 'Unknown error')}")
                    break
                elif training_status not in ["running", "generating_policy", "training"]:
                    print(f"⚠️ Unexpected status: {training_status}")
                
                # Wait before next check
                await asyncio.sleep(2)
                
                # Timeout after 5 minutes
                if time.time() - start_time > 300:
                    print(f"⏱️ Test timeout after 5 minutes")
                    break
            
        except Exception as e:
            print(f"❌ Test case failed: {e}")
    
    print("\n🏁 Training pipeline test completed!")

async def test_scene_availability():
    """Test that all scenes are available for training"""
    
    print("\n🔍 Testing Scene Availability...")
    
    try:
        response = requests.get("http://localhost:8000/api/scenes", timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to get scenes: {response.status_code}")
            return
        
        scenes_data = response.json()
        scenes = scenes_data.get("scenes", {})
        
        print(f"📁 Found {sum(len(category_scenes) for category_scenes in scenes.values())} scenes:")
        
        for category, category_scenes in scenes.items():
            print(f"  {category.title()} ({len(category_scenes)} scenes):")
            for scene in category_scenes:
                scene_name = f"{category}/{scene['name']}"
                xml_path = scene['xml_path']
                
                # Check if XML file exists
                full_path = Path(f"../{xml_path}")
                exists = "✅" if full_path.exists() else "❌"
                print(f"    {exists} {scene_name} -> {xml_path}")
        
        print("✅ Scene availability check completed!")
        
    except Exception as e:
        print(f"❌ Scene availability test failed: {e}")

async def main():
    """Run all tests"""
    
    print("🚀 TabRL Training Pipeline Tests")
    print("=" * 40)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("❌ Backend not responding properly")
            return
    except:
        print("❌ Backend is not running! Please start with: cd backend && uv run python app.py")
        return
    
    # Test scene availability
    await test_scene_availability()
    
    # Test training pipeline
    await test_training_api()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
