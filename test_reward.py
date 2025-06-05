#!/usr/bin/env python3
"""
Quick test of the training API with Claude Opus reward function
"""

import requests
import json
import time

# Claude Opus reward function for dog trick
REWARD_CODE = '''
def reward_fn(state, action):
    # Sparse reward for completing the full trick pose
    
    # Check if sitting position is achieved
    torso_height = state.x.pos[0, 2]
    is_sitting = (0.3 < torso_height < 0.4)
    
    # Check if properly upright with slight backward lean
    torso_pitch = state.x.rot[0, 1]
    is_upright = (-0.3 < torso_pitch < -0.1)
    
    # Check head tilt
    torso_roll = state.x.rot[0, 0]
    has_head_tilt = (0.1 < abs(torso_roll) < 0.25)
    
    # Check paw raise (front leg height difference)
    if len(state.x.pos) > 2:
        left_front_height = state.x.pos[1, 2]
        right_front_height = state.x.pos[2, 2]
        height_diff = abs(left_front_height - right_front_height)
        has_paw_raised = height_diff > 0.05
    else:
        has_paw_raised = False
    
    # Check stability (low velocity)
    is_stable = jnp.sum(state.xd.vel[0]**2) < 0.1
    
    # Give reward only if all conditions are met
    if is_sitting and is_upright and has_head_tilt and has_paw_raised and is_stable:
        reward = 10.0
    else:
        # Small penalty to encourage action
        reward = -0.01
    
    return reward
'''

def test_training():
    # Start training
    print("🚀 Starting training with Claude Opus reward...")
    
    payload = {
        "reward_code": REWARD_CODE,
        "reward_name": "Dog Trick Pose",
        "scene_name": "locomotion/Go1JoystickFlatTerrain",
        "num_timesteps": 50000  # Quick test - 1 minute
    }
    
    response = requests.post("http://localhost:8000/api/training/start", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to start training: {response.text}")
        return
    
    result = response.json()
    job_id = result["job_id"]
    print(f"✅ Training started! Job ID: {job_id}")
    
    # Poll for status
    print("\n📊 Polling training status...")
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        status_response = requests.get(f"http://localhost:8000/api/training/status/{job_id}")
        if status_response.status_code != 200:
            print(f"❌ Status check failed: {status_response.text}")
            break
        
        status = status_response.json()
        runtime = status.get("runtime", 0)
        progress = status.get("progress", 0)
        
        print(f"⏱️  Runtime: {runtime:.1f}s | Progress: {progress}% | Status: {status['status']}")
        
        if status["status"] == "COMPLETED":
            print("\n🎉 Training completed!")
            result = status["result"]
            print(f"📈 Final Return: {result['final_return']:.2f}")
            print(f"🎬 Video: {result['video_path']}")
            print(f"💾 Model: {result['model_path']}")
            break
        elif status["status"] == "FAILED":
            print(f"\n❌ Training failed: {status.get('error', 'Unknown error')}")
            break

if __name__ == "__main__":
    test_training()
