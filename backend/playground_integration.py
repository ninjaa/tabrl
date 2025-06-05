"""
MuJoCo Playground Integration for TabRL
Wraps MuJoCo Playground environments with custom reward functions
"""

import jax
from jax import numpy as jp
from mujoco_playground import registry
from mujoco_playground import wrapper
from brax.envs.base import State
from typing import Dict, Any, Callable, Optional, Union
import numpy as np
from pathlib import Path


class TabRLPlaygroundEnvironment:
    """
    Wrapper for MuJoCo Playground environments that allows:
    1. Loading any playground locomotion environment 
    2. Injecting custom LLM-generated reward functions
    3. Compatible with Brax training pipeline
    """
    
    def __init__(
        self, 
        environment_name: str,
        reward_function_code: str,
        episode_length: int = 1000,
        auto_reset: bool = True
    ):
        self.environment_name = environment_name
        self.reward_function_code = reward_function_code
        self.episode_length = episode_length
        self.auto_reset = auto_reset
        
        # Load the base playground environment
        print(f"🤖 Loading playground environment: {environment_name}")
        self.base_env = registry.locomotion.load(environment_name)
        
        # Get environment specs
        self.action_size = self.base_env.action_size
        self.observation_size = self.base_env.observation_size
        
        # Handle dictionary observations (take 'state' key for policy)
        if isinstance(self.observation_size, dict):
            self.policy_obs_size = self.observation_size['state'][0]
            print(f"   Using 'state' observation: {self.policy_obs_size}D")
        else:
            self.policy_obs_size = self.observation_size
            
        print(f"   Action size: {self.action_size}D")
        print(f"   Total observations: {self.observation_size}")
        
        # Compile custom reward function
        self._compile_reward_function()
        
        # Track episode steps
        self.current_step = 0
    
    def _compile_reward_function(self):
        """Compile the LLM-generated reward function for JAX"""
        try:
            # Create safe execution environment with semantic functions
            reward_globals = {
                'jp': jp,
                'jax': jax,
                # Semantic API functions
                'get_joint_angle': self._get_joint_angle,
                'get_joint_velocity': self._get_joint_velocity,
                'get_body_position': self._get_body_position,
                'get_body_velocity': self._get_body_velocity,
                'get_contact_force': self._get_contact_force,
                'get_base_orientation': self._get_base_orientation,
                'get_foot_contact': self._get_foot_contact,
            }
            
            # Execute the reward function code
            exec(self.reward_function_code, reward_globals)
            
            # Extract the reward function
            if 'compute_reward' in reward_globals:
                self.custom_reward_function = reward_globals['compute_reward']
                print("✅ Custom reward function compiled successfully")
            else:
                raise ValueError("No 'compute_reward' function found in reward code")
                
        except Exception as e:
            print(f"⚠️ Error compiling reward function: {e}")
            # Fallback to default reward
            self.custom_reward_function = None
    
    def _get_observation(self, state_dict: Dict[str, jp.ndarray]) -> jp.ndarray:
        """Extract policy observation from state dictionary"""
        if isinstance(state_dict, dict):
            # Use 'state' key for policy (non-privileged observations)
            return state_dict.get('state', state_dict.get('observation', jp.zeros(self.policy_obs_size)))
        else:
            return state_dict
    
    def _compute_reward(self, state, action, next_state) -> float:
        """Compute reward using custom function or fall back to original"""
        
        if self.custom_reward_function is not None:
            try:
                # Get MJX data from playground state structure
                mjx_data = next_state.data if hasattr(next_state, 'data') else next_state
                custom_reward = self.custom_reward_function(mjx_data)
                return float(custom_reward)
            except Exception as e:
                print(f"⚠️ Custom reward function error: {e}")
                # Fall back to original reward
                return float(next_state.reward)
        else:
            # Use original environment reward
            return float(next_state.reward)
    
    # Semantic API functions for reward design
    def _get_joint_angle(self, mjx_data, joint_name: str) -> float:
        """Get joint angle by name"""
        # TODO: Implement joint name lookup via model
        # For now, return a reasonable joint angle (skip base position/orientation)
        joint_start = 7 if len(mjx_data.qpos) > 7 else 0  # Skip floating base
        joint_idx = min(joint_start, len(mjx_data.qpos) - 1)
        return mjx_data.qpos[joint_idx] if len(mjx_data.qpos) > joint_idx else 0.0
    
    def _get_joint_velocity(self, mjx_data, joint_name: str) -> float:
        """Get joint velocity by name"""
        joint_start = 6 if len(mjx_data.qvel) > 6 else 0  # Skip base velocity
        joint_idx = min(joint_start, len(mjx_data.qvel) - 1)
        return mjx_data.qvel[joint_idx] if len(mjx_data.qvel) > joint_idx else 0.0
    
    def _get_body_position(self, mjx_data, body_name: str = 'torso') -> jp.ndarray:
        """Get body position by name (default to base/torso)"""
        # Base position is typically first 3 elements of qpos
        return mjx_data.qpos[:3] if len(mjx_data.qpos) >= 3 else jp.zeros(3)
    
    def _get_body_velocity(self, mjx_data, body_name: str = 'torso') -> jp.ndarray:
        """Get body velocity by name"""
        return mjx_data.qvel[:3] if len(mjx_data.qvel) >= 3 else jp.zeros(3)
    
    def _get_base_orientation(self, mjx_data) -> jp.ndarray:
        """Get base orientation (quaternion)"""
        # Orientation is typically qpos[3:7] for floating base
        return mjx_data.qpos[3:7] if len(mjx_data.qpos) >= 7 else jp.array([1, 0, 0, 0])
    
    def _get_contact_force(self, mjx_data, body_name: str) -> float:
        """Get contact force for a body"""
        # Check if contact forces are available
        if hasattr(mjx_data, 'contact') and hasattr(mjx_data.contact, 'force'):
            # Return sum of contact forces magnitude
            forces = mjx_data.contact.force
            return jp.sum(jp.linalg.norm(forces, axis=-1)) if len(forces) > 0 else 0.0
        return 0.0
    
    def _get_foot_contact(self, mjx_data, foot_name: str) -> bool:
        """Check if foot is in contact with ground"""
        # Basic contact detection based on contact forces
        contact_force = self._get_contact_force(mjx_data, foot_name)
        return contact_force > 0.1  # Threshold for contact
    
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment"""
        self.current_step = 0
        
        # Reset base environment
        base_state = self.base_env.reset(rng)
        
        # Extract policy observation
        obs = self._get_observation(base_state.obs)
        
        # Compute custom reward if available
        reward = self._compute_reward(None, None, base_state)
        
        # Create compatible state - use base_state directly as pipeline_state
        state = State(
            pipeline_state=base_state,  # Use full state as pipeline state
            obs=obs,
            reward=reward,
            done=jp.array(False),
            metrics={'original_reward': base_state.reward, 'custom_reward': reward},
            info=getattr(base_state, 'info', {})
        )
        
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment"""
        self.current_step += 1
        
        # Get the base state from pipeline_state
        base_state = state.pipeline_state
        
        # Step base environment
        base_next_state = self.base_env.step(base_state, action)
        
        # Extract policy observation
        obs = self._get_observation(base_next_state.obs)
        
        # Compute custom reward
        reward = self._compute_reward(state, action, base_next_state)
        
        # Check if done (episode length or environment termination)
        done = jp.logical_or(
            base_next_state.done,
            self.current_step >= self.episode_length
        )
        
        # Create compatible state
        new_state = State(
            pipeline_state=base_next_state,  # Store full base state
            obs=obs,
            reward=reward,
            done=done,
            metrics={
                'original_reward': base_next_state.reward, 
                'custom_reward': reward,
                'step': self.current_step
            },
            info=getattr(base_next_state, 'info', {})
        )
        
        return new_state


def create_playground_env(
    environment_name: str,
    reward_code: str,
    config: Optional[Dict[str, Any]] = None
) -> TabRLPlaygroundEnvironment:
    """
    Factory function to create TabRL playground environment
    
    Args:
        environment_name: Playground environment name (e.g., 'Go1JoystickFlatTerrain')
        reward_code: LLM-generated reward function code
        config: Environment configuration overrides
    
    Returns:
        Configured TabRL playground environment
    """
    # Default config
    default_config = {
        'episode_length': 1000,
        'auto_reset': True
    }
    
    if config:
        default_config.update(config)
    
    # Create environment
    env = TabRLPlaygroundEnvironment(
        environment_name=environment_name,
        reward_function_code=reward_code,
        **default_config
    )
    
    return env


def test_playground_integration():
    """Test function to validate playground integration"""
    
    # Simple locomotion reward function for testing
    test_reward_code = """
def compute_reward(mjx_data):
    # Forward progress reward
    base_pos = get_body_position(mjx_data)
    forward_progress = base_pos[0]  # X-axis progress
    
    # Speed reward
    base_vel = get_body_velocity(mjx_data)
    speed_reward = base_vel[0] * 0.5  # Forward velocity
    
    # Orientation penalty (keep upright)
    orientation = get_base_orientation(mjx_data)
    upright_reward = orientation[0] * 2.0  # w component of quaternion
    
    return forward_progress + speed_reward + upright_reward
"""
    
    try:
        print("=== Testing Playground Integration ===")
        
        # Create environment
        env = create_playground_env(
            environment_name="Go1JoystickFlatTerrain",
            reward_code=test_reward_code
        )
        
        # Test reset
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        
        print(f"✅ Reset successful")
        print(f"   Observation shape: {state.obs.shape}")
        print(f"   Action size: {env.action_size}")
        print(f"   Original reward: {state.metrics['original_reward']}")
        print(f"   Custom reward: {state.metrics['custom_reward']}")
        
        # Test step
        action = jp.zeros(env.action_size)
        new_state = env.step(state, action)
        
        print(f"✅ Step successful")
        print(f"   New custom reward: {new_state.metrics['custom_reward']}")
        print(f"   Done: {new_state.done}")
        print(f"   Step: {new_state.metrics['step']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_playground_integration()
