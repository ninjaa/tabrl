"""
Brax Environment Wrapper for TabRL
Integrates our MuJoCo scenes with Brax training pipeline
"""

import jax
from jax import numpy as jp
import brax
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.mjx.base import State as MjxState
import mujoco
from mujoco import mjx
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import numpy as np

class TabRLBraxEnvironment(PipelineEnv):
    """
    Brax environment wrapper for TabRL MuJoCo scenes
    Loads our XML files and integrates LLM-generated reward functions
    """
    
    def __init__(
        self, 
        scene_xml_path: str,
        reward_function_code: str,
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True
    ):
        self.scene_xml_path = scene_xml_path
        self.reward_function_code = reward_function_code
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.auto_reset = auto_reset
        
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Create MJX model for JAX acceleration
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # Set up dimensions
        self.observation_size = self.mj_model.nq + self.mj_model.nv  # positions + velocities
        self.action_size = self.mj_model.nu  # actuators
        
        # Compile reward function
        self._compile_reward_function()
        
        print(f"🤖 Brax environment loaded: {self.action_size}D action, {self.observation_size}D observation")
        print(f"📁 Scene: {Path(scene_xml_path).name}")
    
    def _compile_reward_function(self):
        """Compile the LLM-generated reward function for JAX"""
        try:
            # Create safe execution environment with Brax-compatible functions
            reward_globals = {
                'jp': jp,
                'jax': jax,
                'mujoco': mujoco,
                'mjx': mjx,
                # Semantic API functions (to be implemented)
                'get_joint_angle': self._get_joint_angle,
                'get_joint_velocity': self._get_joint_velocity,
                'get_body_position': self._get_body_position,
                'get_body_velocity': self._get_body_velocity,
                'get_contact_force': self._get_contact_force,
            }
            
            # Execute the reward function code
            exec(self.reward_function_code, reward_globals)
            
            # Extract the reward function
            if 'compute_reward' in reward_globals:
                self.reward_function = reward_globals['compute_reward']
                print("✅ Reward function compiled successfully")
            else:
                raise ValueError("No 'compute_reward' function found in reward code")
                
        except Exception as e:
            print(f"⚠️ Error compiling reward function: {e}")
            # Fallback to simple reward
            self.reward_function = self._simple_reward
    
    def _simple_reward(self, state: MjxState) -> float:
        """Simple fallback reward function"""
        # Encourage staying upright and moving forward
        base_pos = state.qpos[:3] if len(state.qpos) >= 3 else jp.zeros(3)
        forward_reward = base_pos[0] if len(base_pos) > 0 else 0.0
        
        # Small action penalty
        action_penalty = jp.sum(state.qfrc_applied**2) * 0.01
        
        return forward_reward - action_penalty
    
    # Semantic API functions for reward design
    def _get_joint_angle(self, state: MjxState, joint_name: str) -> float:
        """Get joint angle by name from Brax state"""
        # TODO: Implement joint name lookup in MJX
        # For now, return first joint angle
        return state.qpos[0] if len(state.qpos) > 0 else 0.0
    
    def _get_joint_velocity(self, state: MjxState, joint_name: str) -> float:
        """Get joint velocity by name from Brax state"""
        return state.qvel[0] if len(state.qvel) > 0 else 0.0
    
    def _get_body_position(self, state: MjxState, body_name: str) -> jp.ndarray:
        """Get body position by name from Brax state"""
        # Return base position (first 3 elements typically)
        return state.qpos[:3] if len(state.qpos) >= 3 else jp.zeros(3)
    
    def _get_body_velocity(self, state: MjxState, body_name: str) -> jp.ndarray:
        """Get body velocity by name from Brax state"""
        return state.qvel[:3] if len(state.qvel) >= 3 else jp.zeros(3)
    
    def _get_contact_force(self, state: MjxState, body_name: str) -> float:
        """Get contact force by name from Brax state"""
        # Placeholder - need to implement contact force extraction
        return 0.0
    
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment"""
        # Initialize MJX state
        mjx_data = mjx.make_data(self.mjx_model)
        
        # Set initial configuration
        mjx_data = mjx_data.replace(
            qpos=self.mjx_model.qpos0,
            qvel=jp.zeros_like(mjx_data.qvel),
            time=0.0
        )
        
        # Create observation
        obs = jp.concatenate([mjx_data.qpos, mjx_data.qvel])
        
        # Compute initial reward
        reward = self.reward_function(mjx_data)
        
        # Create Brax state
        state = State(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=jp.array(False),
            metrics={},
            info={}
        )
        
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment"""
        mjx_data = state.pipeline_state
        
        # Apply action
        mjx_data = mjx_data.replace(ctrl=action)
        
        # Step physics for action_repeat times
        for _ in range(self.action_repeat):
            mjx_data = mjx.step(self.mjx_model, mjx_data)
        
        # Create observation
        obs = jp.concatenate([mjx_data.qpos, mjx_data.qvel])
        
        # Compute reward
        reward = self.reward_function(mjx_data)
        
        # Check if done
        done = jp.logical_or(
            mjx_data.time > self.episode_length * self.mjx_model.opt.timestep,
            self._is_terminated(mjx_data)
        )
        
        # Create new state
        state = State(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            metrics={'reward': reward},
            info={}
        )
        
        return state
    
    def _is_terminated(self, mjx_data) -> jax.Array:
        """Check if episode should terminate early"""
        # Basic termination conditions
        
        # Check for NaN or extreme values
        pos_check = jp.any(jp.logical_or(
            jp.isnan(mjx_data.qpos),
            jp.abs(mjx_data.qpos) > 100.0
        ))
        
        vel_check = jp.any(jp.logical_or(
            jp.isnan(mjx_data.qvel),
            jp.abs(mjx_data.qvel) > 1000.0
        ))
        
        return jp.logical_or(pos_check, vel_check)


def create_tabrl_brax_env(
    scene_name: str,
    reward_code: str,
    config: Optional[Dict[str, Any]] = None
) -> TabRLBraxEnvironment:
    """
    Factory function to create TabRL Brax environment
    
    Args:
        scene_name: Scene identifier (e.g., 'locomotion/anybotics_anymal_c')
        reward_code: LLM-generated reward function code
        config: Environment configuration overrides
    
    Returns:
        Configured TabRL Brax environment
    """
    # Build scene path
    scenes_dir = Path(__file__).parent / "../scenes"
    scene_xml_path = scenes_dir / scene_name / "scene.xml"
    
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_xml_path}")
    
    # Default config
    default_config = {
        'episode_length': 1000,
        'action_repeat': 1,
        'auto_reset': True
    }
    
    if config:
        default_config.update(config)
    
    # Create environment
    env = TabRLBraxEnvironment(
        scene_xml_path=str(scene_xml_path),
        reward_function_code=reward_code,
        **default_config
    )
    
    return env


def test_brax_environment():
    """Test function to validate Brax environment setup"""
    
    # Simple reward function for testing
    test_reward_code = """
def compute_reward(state):
    # Simple forward progress reward
    base_pos = state.qpos[:3] if len(state.qpos) >= 3 else jp.zeros(3)
    return base_pos[0]  # X-axis progress
"""
    
    try:
        # Test with ANYmal scene
        env = create_tabrl_brax_env(
            scene_name="locomotion/anybotics_anymal_c",
            reward_code=test_reward_code
        )
        
        # Test reset
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        
        print(f"✅ Reset successful")
        print(f"   Observation shape: {state.obs.shape}")
        print(f"   Action size: {env.action_size}")
        print(f"   Initial reward: {state.reward}")
        
        # Test step
        action = jp.zeros(env.action_size)
        new_state = env.step(state, action)
        
        print(f"✅ Step successful")
        print(f"   New reward: {new_state.reward}")
        print(f"   Done: {new_state.done}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_brax_environment()
