# structured_observations.py

"""
The REAL solution: Provide semantic access to MuJoCo data
so LLM can write meaningful rewards!
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ObservationStructure:
    """Describes what's in the observation vector AND provides semantic access"""
    
    # Flat observation for neural network
    flat_obs: np.ndarray
    
    # Semantic access for reward writing
    qpos_map: Dict[str, int]      # joint_name -> index in qpos
    body_pos_map: Dict[str, int]  # body_name -> body id
    site_pos_map: Dict[str, int]  # site_name -> site id
    sensor_map: Dict[str, int]    # sensor_name -> index in sensordata
    
    # Metadata
    total_dim: int
    components: List[str]

class StructuredMuJoCoEnv:
    """
    Provides BOTH:
    1. Flat observations for neural network (generic)
    2. Structured access for reward function (semantic)
    """
    
    def __init__(self, xml_string):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        # Build semantic maps from XML
        self._build_semantic_maps()
        
    def _build_semantic_maps(self):
        """Extract names and build lookup tables"""
        
        # Joint name -> qpos index
        self.qpos_map = {}
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            qpos_addr = self.model.jnt_qposadr[i]
            self.qpos_map[name] = qpos_addr
            
        # Body name -> body id  
        self.body_map = {}
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            self.body_map[name] = i
            
        # Site name -> site id (attachment points)
        self.site_map = {}
        for i in range(self.model.nsite):
            name = self.model.site(i).name
            self.site_map[name] = i
            
        # Sensor name -> sensor data index
        self.sensor_map = {}
        for i in range(self.model.nsensor):
            name = self.model.sensor(i).name
            self.sensor_map[name] = i
    
    def get_flat_observations(self) -> np.ndarray:
        """For neural network - just concatenate everything"""
        obs = []
        obs.extend(self.data.qpos)
        obs.extend(self.data.qvel)
        obs.extend(self.data.sensordata)
        return np.array(obs, dtype=np.float32)
    
    def get_semantic_state(self) -> Dict:
        """For reward function - organized access"""
        return {
            'qpos': self.data.qpos,
            'qvel': self.data.qvel,
            'body_pos': {name: self.data.body(id).xpos 
                         for name, id in self.body_map.items()},
            'site_pos': {name: self.data.site(id).xpos
                         for name, id in self.site_map.items()},
            'sensors': {name: self.data.sensordata[id]
                       for name, id in self.sensor_map.items()},
            # Convenience accessors
            'get_body_pos': lambda name: self.data.body(self.body_map[name]).xpos,
            'get_joint_angle': lambda name: self.data.qpos[self.qpos_map[name]],
            'get_sensor': lambda name: self.data.sensordata[self.sensor_map[name]]
        }

# =============================================
# What the LLM receives to write rewards
# =============================================

def generate_reward_with_structure(env, task_prompt):
    """LLM gets structure info to write meaningful rewards"""
    
    # Analyze available components
    structure_info = {
        'joints': list(env.qpos_map.keys()),
        'bodies': list(env.body_map.keys()),
        'sites': list(env.site_map.keys()),
        'sensors': list(env.sensor_map.keys())
    }
    
    # LLM prompt includes this structure
    llm_context = f"""
    You are writing a reward function for: {task_prompt}
    
    Available components:
    - Joints: {structure_info['joints']}
    - Bodies: {structure_info['bodies']}  
    - Sites: {structure_info['sites']} (attachment points like 'gripper_tip')
    - Sensors: {structure_info['sensors']}
    
    You can access these in the reward function using:
    - state['get_body_pos']('body_name') -> [x, y, z]
    - state['get_joint_angle']('joint_name') -> angle in radians
    - state['get_sensor']('sensor_name') -> sensor value
    
    Write a reward function using these semantic names.
    """
    
    return llm_context

# =============================================
# Example: LLM-generated rewards using structure
# =============================================

def pick_and_place_reward_structured(model, data, state):
    """LLM can write THIS because it knows the structure"""
    
    # Access by semantic names!
    gripper_pos = state['get_site_pos']('gripper_tip')
    object_pos = state['get_body_pos']('red_block')
    gripper_open = state['get_joint_angle']('gripper_joint')
    touching = state['get_sensor']('gripper_touch_sensor')
    
    # Meaningful reward using named components
    distance = np.linalg.norm(gripper_pos - object_pos)
    
    reward = 0.0
    reward -= distance  # Get close
    
    if touching > 0.5 and gripper_open < 0.1:  # Grasping
        reward += 10.0
        
    if object_pos[2] > 0.1:  # Lifted off ground
        reward += 5.0
        
    return reward

def humanoid_walk_reward_structured(model, data, state):
    """LLM knows which joints and bodies exist"""
    
    # Access specific components
    pelvis_height = state['get_body_pos']('pelvis')[2]
    pelvis_velocity = state['qvel'][0]  # Forward velocity
    
    left_foot_contact = state['get_sensor']('left_foot_touch')
    right_foot_contact = state['get_sensor']('right_foot_touch')
    
    # Structured reward
    reward = pelvis_velocity  # Forward progress
    
    if pelvis_height < 0.7:  # Falling
        reward -= 10.0
        
    # Encourage alternating foot contacts
    if left_foot_contact > 0 and right_foot_contact == 0:
        reward += 0.1
    elif right_foot_contact > 0 and left_foot_contact == 0:  
        reward += 0.1
        
    return reward

# =============================================
# The Complete System Design
# =============================================

"""
1. OBSERVATION EXTRACTION:
   - Still use flat observations for neural network (simple, generic)
   - obs = [qpos, qvel, sensordata] concatenated

2. REWARD FUNCTION:
   - LLM gets semantic structure from XML parsing
   - Can write rewards using meaningful names
   - No guessing about indices!

3. TRAINING:
   - Neural network trains on flat observations
   - Reward calculated using semantic access
   - Best of both worlds!
"""

# What Modal actually runs
class TabRLTrainingEnv(gym.Env):
    def __init__(self, xml_string, reward_code):
        super().__init__()
        
        # Setup MuJoCo
        self.env = StructuredMuJoCoEnv(xml_string)
        
        # Compile reward function
        exec(reward_code, globals())
        self.reward_fn = globals()['compute_reward']
        
        # Spaces (using flat observations)
        obs_dim = len(self.env.get_flat_observations())
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,))
        self.action_space = spaces.Box(-1, 1, (self.env.model.nu,))
        
    def reset(self):
        self.env.reset()
        return self.env.get_flat_observations()
        
    def step(self, action):
        # Apply action
        self.env.data.ctrl[:] = action
        self.env.step()
        
        # Get flat observation for neural network
        obs = self.env.get_flat_observations()
        
        # Get semantic state for reward calculation  
        semantic_state = self.env.get_semantic_state()
        
        # Calculate reward using semantic access
        reward = self.reward_fn(self.env.model, self.env.data, semantic_state)
        
        done = False
        
        return obs, reward, done, {}

# =============================================
# What gets sent to the LLM
# =============================================

EXAMPLE_LLM_REQUEST = {
    "task": "Make the robot pick up the red block",
    "scene_structure": {
        "robot_type": "manipulator",
        "joints": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper_joint"],
        "bodies": ["base", "link1", "link2", "link3", "link4", "link5", "link6", "gripper", "red_block", "blue_block", "table"],
        "sites": ["gripper_tip", "camera_mount"],
        "sensors": ["gripper_touch_sensor", "joint_torque_1", "joint_torque_2"]
    },
    "example_access": """
    # You can write rewards like this:
    gripper_pos = state['get_site_pos']('gripper_tip')
    red_block_pos = state['get_body_pos']('red_block')
    distance = np.linalg.norm(gripper_pos - red_block_pos)
    """
}