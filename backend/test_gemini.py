#!/usr/bin/env python3
"""
Test Gemini endpoint specifically
"""

import requests
import json

def test_gemini_endpoint():
    """Test if Gemini is returning real results or fallback data"""
    
    print("🔵 Testing Gemini Pro 2.5 endpoint...")
    
    # Test payload - using correct parameter names
    payload = {
        "task_description": "make the robot do a backflip",
        "model_id": "locomotion/Go1JoystickFlatTerrain",  # This is the scene_name
        "llm_model": "gemini/gemini-2.5-pro-preview-06-05",  # Gemini model
        "num_approaches": 3
    }
    
    print(f"Task: {payload['task_description']}")
    print(f"LLM Model: {payload['llm_model']}")
    print(f"Model ID (Scene): {payload['model_id']}")
    
    # Call the approaches endpoint
    response = requests.post("http://localhost:8000/api/training/approaches", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    approaches = result.get("approaches", [])
    
    print(f"\n✅ Got {len(approaches)} approaches from Gemini\n")
    
    # Check if we're getting real results or mock data
    for i, approach in enumerate(approaches):
        print(f"--- Approach {i+1}: {approach['name']} ---")
        print(f"Reward Code Length: {len(approach.get('reward_code', ''))}")
        
        # Extract a snippet of the reward code
        reward_code = approach['reward_code']
        lines = reward_code.split('\n')
        
        # Look for signs of mock data
        if "get_joint_angle" in reward_code and "shoulder_pan_joint" in reward_code:
            print("⚠️  Contains mock data patterns (shoulder_pan_joint)")
        elif "backflip" in reward_code.lower() or "flip" in reward_code.lower():
            print("✅ Contains backflip-specific logic")
        else:
            print("🤔 Unclear if this is task-specific")
        
        # Show first 10 lines
        print("\nFirst 10 lines of reward code:")
        for line in lines[:10]:
            print(f"  {line}")
        
        print()
    
    # Let's also test the policy generation endpoint directly
    print("\n🔬 Testing direct policy generation endpoint...")
    
    policy_payload = {
        "prompt": "make the robot do a backflip",
        "scene_name": "locomotion/Go1JoystickFlatTerrain", 
        "model": "gemini/gemini-2.5-pro-preview-06-05",
        "temperature": 0.7
    }
    
    policy_response = requests.post("http://localhost:8000/api/policy/generate", json=policy_payload)
    
    if policy_response.status_code == 200:
        policy_result = policy_response.json()
        print(f"✅ Policy generation successful")
        print(f"Approaches: {len(policy_result.get('reward_functions', []))}")
        
        # Check first reward function
        if policy_result.get('reward_functions'):
            first_rf = policy_result['reward_functions'][0]
            print(f"\nFirst reward function name: {first_rf.get('name')}")
            code = first_rf.get('code', '')
            if 'backflip' in code.lower() or 'flip' in code.lower():
                print("✅ Contains backflip-specific logic")
            else:
                print("⚠️  May be using default/mock logic")
    else:
        print(f"❌ Policy generation failed: {policy_response.status_code}")
        print(policy_response.text)

if __name__ == "__main__":
    test_gemini_endpoint()
