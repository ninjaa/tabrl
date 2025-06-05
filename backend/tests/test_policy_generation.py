#!/usr/bin/env python3
"""
Test suite for policy generation debugging
"""
import asyncio
import json
from pathlib import Path
from scene_parser import parse_scene_xml, generate_llm_context
from inference import InferenceEngine

async def test_scene_parsing():
    """Test 1: Scene parsing works correctly"""
    print("🔍 Test 1: Scene Parsing")
    
    # Find the scene XML file
    scenes_dir = Path(__file__).parent.parent / "scenes"
    scene_path = scenes_dir / "manipulation/universal_robots_ur5e"
    
    xml_files = ["scene.xml", "robot.xml", "ur5e.xml"]
    xml_file = None
    for xml_name in xml_files:
        if (scene_path / xml_name).exists():
            xml_file = scene_path / xml_name
            break
    
    if not xml_file:
        print("❌ No XML file found")
        return None
    
    print(f"✅ Found XML: {xml_file}")
    
    # Parse the scene
    try:
        scene_structure = parse_scene_xml(str(xml_file))
        print(f"✅ Scene parsed: {scene_structure.nu} actuators, {scene_structure.nq} joints")
        print(f"   Joints: {scene_structure.joints[:3]}...")  # First 3 joints
        return scene_structure
    except Exception as e:
        print(f"❌ Scene parsing failed: {e}")
        return None

async def test_llm_context_generation(scene_structure):
    """Test 2: LLM context generation"""
    print("\n🔍 Test 2: LLM Context Generation")
    
    if not scene_structure:
        print("❌ No scene structure to test with")
        return None
    
    try:
        llm_context = generate_llm_context(scene_structure, "test task")
        print(f"✅ LLM context generated: {len(llm_context)} characters")
        print(f"   Preview: {llm_context[:200]}...")
        return llm_context
    except Exception as e:
        print(f"❌ LLM context generation failed: {e}")
        return None

async def test_structured_policy_generation(scene_structure, llm_context):
    """Test 3: Structured policy generation"""
    print("\n🔍 Test 3: Structured Policy Generation")
    
    if not scene_structure or not llm_context:
        print("❌ Missing prerequisites")
        return None
    
    try:
        inference_engine = InferenceEngine()
        
        # Test the raw LLM response
        policy_data = await inference_engine.generate_structured_policy(
            prompt="test task",
            obs_space=scene_structure.nq + scene_structure.nv,
            action_space=scene_structure.nu,
            scene_context=llm_context,
            model="claude-3-5-sonnet-20241022"
        )
        
        print(f"✅ Raw LLM response keys: {list(policy_data.keys())}")
        print(f"   Policy code length: {len(policy_data.get('policy_code', ''))}")
        print(f"   Reward functions type: {type(policy_data.get('reward_functions'))}")
        print(f"   Reward functions content: {policy_data.get('reward_functions')}")
        
        return policy_data
    except Exception as e:
        print(f"❌ Structured policy generation failed: {e}")
        return None

async def test_json_parsing(policy_data):
    """Test 4: JSON parsing logic"""
    print("\n🔍 Test 4: JSON Parsing Logic")
    
    if not policy_data:
        print("❌ No policy data to test with")
        return
    
    # Test our JSON parsing logic
    reward_functions = policy_data.get("reward_functions")
    print(f"Original reward_functions type: {type(reward_functions)}")
    print(f"Original reward_functions value: {reward_functions}")
    
    # Apply our parsing logic
    if isinstance(reward_functions, str):
        print("🔧 Attempting to parse JSON string...")
        try:
            parsed = json.loads(reward_functions)
            print(f"✅ JSON parsing successful: {len(parsed)} reward functions")
            for i, rf in enumerate(parsed):
                print(f"   Reward {i+1}: {rf.get('name', 'unnamed')} ({rf.get('type', 'unknown')})")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"   Raw string: {reward_functions[:200]}...")
    elif isinstance(reward_functions, list):
        print(f"✅ Already a list with {len(reward_functions)} items")
        for i, rf in enumerate(reward_functions):
            print(f"   Reward {i+1}: {rf.get('name', 'unnamed')} ({rf.get('type', 'unknown')})")
    else:
        print(f"❌ Unexpected type: {type(reward_functions)}")

async def main():
    """Run all tests"""
    print("🧪 Policy Generation Debug Tests\n")
    
    # Test 1: Scene parsing
    scene_structure = await test_scene_parsing()
    
    # Test 2: LLM context generation  
    llm_context = await test_llm_context_generation(scene_structure)
    
    # Test 3: Structured policy generation
    policy_data = await test_structured_policy_generation(scene_structure, llm_context)
    
    # Test 4: JSON parsing
    await test_json_parsing(policy_data)
    
    print("\n🏁 Tests complete!")

if __name__ == "__main__":
    asyncio.run(main())
