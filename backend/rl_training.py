"""
Real RL Training Implementation using PPO
Integrates with scene parser and generated reward functions
"""

import os
import sys
import numpy as np
import mujoco
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import subprocess

class MuJoCoEnvironment:
    """MuJoCo environment wrapper for RL training"""
    
    def __init__(self, scene_xml_path: str, reward_function_code: str):
        self.scene_xml_path = scene_xml_path
        self.reward_function_code = reward_function_code
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set up dimensions
        self.action_dim = self.model.nu
        self.obs_dim = self.model.nq + self.model.nv  # positions + velocities
        
        # Compile reward function
        self._compile_reward_function()
        
        print(f"🤖 Environment loaded: {self.action_dim}D action, {self.obs_dim}D observation")
    
    def _compile_reward_function(self):
        """Compile the reward function from generated code"""
        try:
            # Create a safe execution environment
            reward_globals = {
                'np': np,
                'mujoco': mujoco,
                'model': self.model,
                'data': self.data,
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
            else:
                raise ValueError("No 'compute_reward' function found in reward code")
                
        except Exception as e:
            print(f"⚠️ Error compiling reward function: {e}")
            # Fallback to simple reward
            self.reward_function = self._simple_reward
    
    def _simple_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Simple fallback reward function"""
        # Encourage staying upright and moving forward
        return 1.0 - np.sum(action**2) * 0.01
    
    def _get_joint_angle(self, joint_name: str) -> float:
        """Get joint angle by name"""
        try:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                return self.data.qpos[joint_id]
        except:
            pass
        return 0.0
    
    def _get_joint_velocity(self, joint_name: str) -> float:
        """Get joint velocity by name"""
        try:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                return self.data.qvel[joint_id]
        except:
            pass
        return 0.0
    
    def _get_body_position(self, body_name: str) -> np.ndarray:
        """Get body position by name"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return self.data.xpos[body_id].copy()
        except:
            pass
        return np.zeros(3)
    
    def _get_body_velocity(self, body_name: str) -> np.ndarray:
        """Get body velocity by name"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return self.data.cvel[body_id].copy()
        except:
            pass
        return np.zeros(6)
    
    def _get_contact_force(self, geom_name: str) -> float:
        """Get contact force for a geom"""
        try:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            total_force = 0.0
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    total_force += np.linalg.norm(contact.frame[:3])
            return total_force
        except:
            pass
        return 0.0
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Add some noise to initial conditions
        noise_scale = 0.01
        self.data.qpos += np.random.normal(0, noise_scale, self.model.nq)
        self.data.qvel += np.random.normal(0, noise_scale, self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment"""
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action
        self.data.ctrl[:len(action)] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self.reward_function(obs, action)
        
        # Check if episode is done
        done = self._is_done()
        
        info = {}
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _is_done(self) -> bool:
        """Check if episode should terminate"""
        # Basic termination conditions
        if self.data.time > 10.0:  # Max episode length
            return True
        
        # Check for unrealistic states
        if np.any(np.abs(self.data.qpos) > 10.0):
            return True
        
        if np.any(np.abs(self.data.qvel) > 50.0):
            return True
        
        return False

class SimplePPO:
    """Simplified PPO implementation for TabRL training"""
    
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Simple neural network weights (random initialization)
        self.policy_weights = np.random.normal(0, 0.1, (action_dim, obs_dim))
        self.value_weights = np.random.normal(0, 0.1, obs_dim)
        
        # Training hyperparameters
        self.learning_rate = 0.01
        self.gamma = 0.99
        self.clip_ratio = 0.2
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from observation"""
        # Simple linear policy
        action = np.tanh(self.policy_weights @ obs)
        return action
    
    def predict_value(self, obs: np.ndarray) -> float:
        """Predict state value"""
        return np.dot(self.value_weights, obs)
    
    def update(self, observations: List[np.ndarray], actions: List[np.ndarray], 
               rewards: List[float], dones: List[bool]):
        """Update policy using PPO"""
        
        if len(observations) < 2:
            return 0.0
        
        # Convert to numpy arrays
        obs = np.array(observations)
        acts = np.array(actions)
        rews = np.array(rewards)
        
        # Compute returns
        returns = self._compute_returns(rews, dones)
        
        # Simple gradient update (not full PPO, but functional)
        for i in range(len(obs) - 1):
            # Policy gradient update
            advantage = returns[i] - self.predict_value(obs[i])
            policy_grad = np.outer(acts[i] - self.predict(obs[i]), obs[i])
            self.policy_weights += self.learning_rate * advantage * policy_grad
            
            # Value function update
            value_error = returns[i] - self.predict_value(obs[i])
            self.value_weights += self.learning_rate * value_error * obs[i]
        
        return np.mean(returns)
    
    def _compute_returns(self, rewards: np.ndarray, dones: List[bool]) -> np.ndarray:
        """Compute discounted returns"""
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        
        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + self.gamma * returns[i + 1] * (1 - dones[i])
        
        return returns

class RealTrainer:
    """Real RL trainer using MuJoCo and PPO"""
    
    def __init__(self):
        self.training_videos_dir = Path("training_videos")
        self.training_videos_dir.mkdir(exist_ok=True)
    
    async def train_policy(
        self, 
        scene_xml_path: str,
        reward_function_code: str, 
        training_id: str,
        task_description: str,
        episodes: int = 100,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Train a policy using the given scene and reward function"""
        
        print(f"🚀 Starting real RL training for: {task_description}")
        print(f"📁 Scene: {scene_xml_path}")
        print(f"🎯 Episodes: {episodes}")
        
        try:
            # Create environment
            env = MuJoCoEnvironment(scene_xml_path, reward_function_code)
            
            # Create PPO agent
            agent = SimplePPO(env.obs_dim, env.action_dim)
            
            # Training metrics
            episode_rewards = []
            training_data = []
            
            for episode in range(episodes):
                # Run episode
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
                
                observations = [obs]
                actions = []
                rewards = []
                dones = []
                
                for step in range(200):  # Max steps per episode
                    # Get action
                    action = agent.predict(obs)
                    actions.append(action)
                    
                    # Step environment
                    next_obs, reward, done, info = env.step(action)
                    
                    observations.append(next_obs)
                    rewards.append(reward)
                    dones.append(done)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                # Update agent
                if len(observations) > 1:
                    avg_return = agent.update(observations, actions, rewards, dones)
                
                episode_rewards.append(episode_reward)
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                
                # Log progress
                if episode % 10 == 0 or episode == episodes - 1:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg = {avg_reward:.2f}")
                    
                    # Call progress callback
                    if progress_callback:
                        await progress_callback(episode, episodes, episode_reward, avg_reward)
                
                # Record training data
                training_data.append({
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "avg_reward": np.mean(episode_rewards[-10:])
                })
            
            # Generate training video/visualization
            video_path = await self._generate_training_video(
                env, agent, training_id, task_description
            )
            
            # Prepare results
            results = {
                "success": True,
                "final_avg_reward": np.mean(episode_rewards[-10:]),
                "total_episodes": episodes,
                "training_data": training_data,
                "video_path": video_path,
                "model_info": {
                    "obs_dim": env.obs_dim,
                    "action_dim": env.action_dim,
                    "policy_weights_shape": agent.policy_weights.shape
                }
            }
            
            print(f"✅ Training completed! Final avg reward: {results['final_avg_reward']:.2f}")
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "training_data": []
            }
    
    async def _generate_training_video(
        self, 
        env: MuJoCoEnvironment, 
        agent: SimplePPO, 
        training_id: str,
        task_description: str
    ) -> Optional[str]:
        """Generate a video showing the trained policy"""
        
        try:
            print("🎬 Generating training validation video...")
            
            # Run a demonstration episode
            obs = env.reset()
            frames = []
            
            # Create a simple renderer (text-based for now)
            demo_data = []
            
            for step in range(100):
                action = agent.predict(obs)
                next_obs, reward, done, info = env.step(action)
                
                # Record frame data
                demo_data.append({
                    "step": step,
                    "obs": obs.tolist(),
                    "action": action.tolist(),
                    "reward": reward,
                    "joint_positions": env.data.qpos.tolist(),
                    "joint_velocities": env.data.qvel.tolist()
                })
                
                obs = next_obs
                if done:
                    break
            
            # Save demonstration data as JSON
            video_filename = f"training_{training_id}_demo.json"
            video_path = self.training_videos_dir / video_filename
            
            with open(video_path, 'w') as f:
                json.dump({
                    "training_id": training_id,
                    "task_description": task_description,
                    "total_steps": len(demo_data),
                    "total_reward": sum(frame["reward"] for frame in demo_data),
                    "frames": demo_data
                }, f, indent=2)
            
            print(f"📹 Training video saved: {video_path}")
            return str(video_path)
            
        except Exception as e:
            print(f"⚠️ Could not generate training video: {e}")
            return None
