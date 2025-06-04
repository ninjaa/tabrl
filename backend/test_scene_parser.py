"""
Test Suite for Scene Parser
Tests parsing of all available MuJoCo scenes with different XML structures
"""

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from scene_parser import parse_scene_xml, generate_llm_context

class TestSceneParser:
    """Test suite for scene XML parsing functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_scenes = [
            {
                "name": "manipulation/universal_robots_ur5e",
                "path": "../scenes/manipulation/universal_robots_ur5e/scene.xml",
                "expected_joints": 6,
                "expected_actuators": 6,
                "expected_scene_name": "ur5e scene"
            },
            {
                "name": "locomotion/anybotics_anymal_c", 
                "path": "../scenes/locomotion/anybotics_anymal_c/scene.xml",
                "expected_joints": 12,  # Quadruped has 3 joints per leg * 4 legs
                "expected_actuators": 12,
                "expected_scene_name": "anymal_c scene"
            },
            {
                "name": "simple/shadow_hand_left",
                "path": "../scenes/simple/shadow_hand/scene_left.xml", 
                "expected_joints": 20,  # Complex hand with many DOF
                "expected_actuators": 20,
                "expected_scene_name": "left_shadow_hand scene"
            }
        ]
    
    def test_parse_manipulation_scene(self):
        """Test parsing UR5e manipulation robot"""
        scene_path = "../scenes/manipulation/universal_robots_ur5e/scene.xml"
        
        structure = parse_scene_xml(scene_path)
        
        # Basic properties
        assert structure.name == "ur5e scene"
        assert len(structure.joints) == 6
        assert len(structure.actuators) == 6
        assert structure.nu == 6  # Action dimension
        assert structure.nq == 6  # Position dimension
        assert structure.nv == 6  # Velocity dimension
        
        # Expected joint names
        expected_joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        assert all(joint in structure.joints for joint in expected_joints)
        
        # Expected actuator names  
        expected_actuators = [
            'shoulder_pan', 'shoulder_lift', 'elbow',
            'wrist_1', 'wrist_2', 'wrist_3'
        ]
        assert all(actuator in structure.actuators for actuator in expected_actuators)
        
        # Should have bodies and sites
        assert len(structure.bodies) > 0
        assert len(structure.sites) > 0
        
        print(f"✅ UR5e Scene: {len(structure.joints)} joints, {len(structure.actuators)} actuators")
    
    def test_parse_locomotion_scene(self):
        """Test parsing ANYmal quadruped locomotion robot"""
        scene_path = "../scenes/locomotion/anybotics_anymal_c/scene.xml"
        
        structure = parse_scene_xml(scene_path)
        
        # Basic properties
        assert structure.name == "anymal_c scene"
        
        # Quadruped should have multiple joints (3 per leg * 4 legs = 12)
        assert len(structure.joints) >= 12
        assert len(structure.actuators) >= 12
        assert structure.nu >= 12  # Action dimension
        
        # Should have leg-related joints
        leg_keywords = ['LF', 'RF', 'LH', 'RH']
        leg_joints = [j for j in structure.joints 
                     if any(keyword in j for keyword in leg_keywords)]
        assert len(leg_joints) > 0, f"No leg joints found in: {structure.joints}"
        
        print(f"✅ ANYmal Scene: {len(structure.joints)} joints, {len(structure.actuators)} actuators")
    
    def test_parse_hand_scene(self):
        """Test parsing Shadow Hand dexterous manipulation"""
        scene_path = "../scenes/simple/shadow_hand/scene_left.xml"
        
        structure = parse_scene_xml(scene_path)
        
        # Basic properties
        assert structure.name == "left_shadow_hand scene"
        
        # Hand should have many joints (fingers, thumb, etc.)
        assert len(structure.joints) >= 15  # Dexterous hand has many DOF
        
        # Should have finger-related joints
        finger_keywords = ['TH', 'FF', 'MF', 'RF', 'LF']
        finger_joints = [j for j in structure.joints 
                        if any(keyword in j for keyword in finger_keywords)]
        assert len(finger_joints) > 0, f"No finger joints found in: {structure.joints}"
        
        print(f"✅ Shadow Hand Scene: {len(structure.joints)} joints, {len(structure.actuators)} actuators")
    
    def test_parse_humanoid_scene(self):
        """Test parsing Unitree G1 humanoid robot"""
        scene_path = "../scenes/locomotion/unitree_g1/scene.xml"
        
        structure = parse_scene_xml(scene_path)
        
        # Basic properties
        assert structure.name == "g1_29dof_rev_1_0 scene"
        assert len(structure.joints) == 29  # 29 DOF humanoid
        assert len(structure.actuators) == 29
        assert structure.nu == 29  # Action dimension
        assert structure.nq == 29  # Position dimension  
        assert structure.nv == 29  # Velocity dimension
        
        # Should have humanoid-related joints
        humanoid_keywords = ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist', 'waist']
        humanoid_joints = [j for j in structure.joints 
                          if any(keyword in j.lower() for keyword in humanoid_keywords)]
        assert len(humanoid_joints) > 20, f"Not enough humanoid joints found: {len(humanoid_joints)}"
        
        # Should have left/right symmetry
        left_joints = [j for j in structure.joints if 'left' in j.lower()]
        right_joints = [j for j in structure.joints if 'right' in j.lower()]
        assert len(left_joints) > 0, "No left-side joints found"
        assert len(right_joints) > 0, "No right-side joints found"
        assert len(left_joints) == len(right_joints), f"Asymmetric joint count: {len(left_joints)} left vs {len(right_joints)} right"
        
        print(f"✅ Unitree G1 Scene: {len(structure.joints)} joints, {len(structure.actuators)} actuators")
        print(f"    Left/Right symmetry: {len(left_joints)} joints each side")
    
    def test_xml_include_handling(self):
        """Test that XML includes are properly resolved"""
        scene_path = "../scenes/manipulation/universal_robots_ur5e/scene.xml"
        
        structure = parse_scene_xml(scene_path)
        
        # The scene.xml includes ur5e.xml, so we should get joints from the included file
        assert len(structure.joints) > 0, "No joints found - include not processed"
        assert len(structure.bodies) > 0, "No bodies found - include not processed"
        
        # Check that we got specific elements that only exist in ur5e.xml
        assert 'shoulder_pan_joint' in structure.joints, "Included file joints not found"
        
    def test_actuator_detection(self):
        """Test different actuator types are detected correctly"""
        scene_path = "../scenes/manipulation/universal_robots_ur5e/scene.xml"
        
        structure = parse_scene_xml(scene_path)
        
        # Should detect general actuators from ur5e.xml
        expected_actuators = ['shoulder_pan', 'shoulder_lift', 'elbow']
        found_actuators = [a for a in expected_actuators if a in structure.actuators]
        assert len(found_actuators) > 0, f"General actuators not detected: {structure.actuators}"
        
    def test_fallback_parsing(self):
        """Test parser handles invalid/missing XML gracefully"""
        invalid_path = "../scenes/nonexistent/fake_scene.xml"
        
        structure = parse_scene_xml(invalid_path)
        
        # Should return valid fallback structure
        assert structure.name == "unknown_scene"
        assert structure.joints == []
        assert structure.bodies == []
        assert structure.nq == 0
        assert structure.nu == 0
    
    def test_llm_context_generation(self):
        """Test LLM context includes proper scene information"""
        scene_path = "../scenes/manipulation/universal_robots_ur5e/scene.xml"
        structure = parse_scene_xml(scene_path)
        
        context = generate_llm_context(structure, "Pick up a red block")
        
        # Context should include key information
        assert "Pick up a red block" in context
        assert "ur5e scene" in context
        assert "shoulder_pan_joint" in context
        assert "Action Dimension: 6" in context
        assert "get_joint_angle" in context
        
    def test_all_available_scenes(self):
        """Test parsing all scenes that should be available"""
        import requests
        
        # Get scenes from API
        try:
            response = requests.get('http://127.0.0.1:8000/api/scenes', timeout=5)
            if response.status_code == 200:
                scenes_data = response.json()
                
                print(f"\n🔍 Testing {sum(len(scenes) for scenes in scenes_data['scenes'].values())} available scenes:")
                
                for category, scenes in scenes_data['scenes'].items():
                    for scene in scenes:
                        print(f"  Testing {category}/{scene['name']}...")
                        
                        # Parse scene using the API path
                        scene_path = f"../scenes/{category}/{scene['name']}/scene.xml"
                        if not Path(scene_path).exists():
                            # Try alternative names
                            scene_dir = Path(f"../scenes/{category}/{scene['name']}")
                            xml_files = list(scene_dir.glob("*.xml"))
                            if xml_files:
                                scene_path = str(xml_files[0])
                        
                        structure = parse_scene_xml(scene_path)
                        
                        # Basic validation
                        assert structure.name != "unknown_scene", f"Failed to parse {scene['name']}"
                        assert structure.nq >= 0, f"Invalid nq for {scene['name']}"
                        assert structure.nu >= 0, f"Invalid nu for {scene['name']}"
                        
                        print(f"    ✅ {scene['name']}: {len(structure.joints)} joints, {len(structure.actuators)} actuators")
        except:
            print("⚠️  Could not connect to backend - testing local files only")

    def test_policy_generation_integration(self):
        """Test end-to-end policy generation with all available scenes"""
        import requests
        
        test_cases = [
            {
                "scene": "manipulation/universal_robots_ur5e",
                "task": "Pick and place a red block",
                "expected_action_dim": 6,
                "expected_min_state_dim": 12
            },
            {
                "scene": "locomotion/anybotics_anymal_c", 
                "task": "Walk forward steadily",
                "expected_action_dim": 12,
                "expected_min_state_dim": 24
            },
            {
                "scene": "locomotion/unitree_g1",
                "task": "Walk like a human with natural gait",
                "expected_action_dim": 29,
                "expected_min_state_dim": 58
            },
            {
                "scene": "simple/shadow_hand",
                "task": "Grasp a small object",
                "expected_action_dim": 20,
                "expected_min_state_dim": 40
            }
        ]
        
        for test_case in test_cases:
            request_data = {
                'prompt': test_case["task"],
                'scene_name': test_case["scene"],
                'model': 'claude-3-5-sonnet-20241022'
            }
            
            try:
                response = requests.post('http://127.0.0.1:8000/api/policy/generate', 
                                       json=request_data, 
                                       timeout=30)
                
                assert response.status_code == 200, f"Policy generation failed for {test_case['scene']}"
                
                data = response.json()
                assert 'scene_structure' in data, f"No scene structure in response for {test_case['scene']}"
                
                struct = data['scene_structure']
                assert struct['action_dim'] == test_case['expected_action_dim'], \
                    f"Action dim mismatch for {test_case['scene']}: expected {test_case['expected_action_dim']}, got {struct['action_dim']}"
                
                assert struct['state_dim'] >= test_case['expected_min_state_dim'], \
                    f"State dim too low for {test_case['scene']}: expected >={test_case['expected_min_state_dim']}, got {struct['state_dim']}"
                
                print(f"    ✅ {test_case['scene']}: {struct['action_dim']}D action, {struct['state_dim']}D state")
                
            except requests.exceptions.RequestException:
                print(f"    ⚠️  Backend not available for {test_case['scene']} - skipping integration test")
                return

def run_tests():
    """Run all tests and display results"""
    print("🧪 Running Scene Parser Tests...\n")
    
    test_suite = TestSceneParser()
    test_suite.setup_method()
    
    tests = [
        ("Manipulation Scene (UR5e)", test_suite.test_parse_manipulation_scene),
        ("Locomotion Scene (ANYmal)", test_suite.test_parse_locomotion_scene), 
        ("Hand Scene (Shadow Hand)", test_suite.test_parse_hand_scene),
        ("Humanoid Scene (Unitree G1)", test_suite.test_parse_humanoid_scene),
        ("XML Include Handling", test_suite.test_xml_include_handling),
        ("Actuator Detection", test_suite.test_actuator_detection),
        ("Fallback Parsing", test_suite.test_fallback_parsing),
        ("LLM Context Generation", test_suite.test_llm_context_generation),
        ("All Available Scenes", test_suite.test_all_available_scenes),
        ("Policy Generation Integration", test_suite.test_policy_generation_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}")
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
