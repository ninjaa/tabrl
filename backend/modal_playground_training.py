"""
Modal GPU Training Service for MuJoCo Playground Locomotion
Based on the official MuJoCo Playground notebook but adapted for TabRL integration.
"""

import modal
import os
from typing import Dict, Any, Optional

# Modal app for training
app = modal.App("tabrl-playground-training")

# Create Modal image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["libegl1-mesa", "libopengl0", "libglu1-mesa", "git"])
    .pip_install("jax==0.4.35")  # Install base JAX first
    .pip_install(
        "jaxlib[cuda12_pip]==0.4.35",  # Then CUDA-enabled jaxlib
        find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    .pip_install([
        "mujoco>=3.0.0",
        "mujoco_mjx>=3.0.0", 
        "brax>=0.11.0",
        "flax",
        "optax",
        "mediapy",
        "matplotlib",
        "numpy",
        "playground",  # MuJoCo Playground environments
    ])
    .env({"MUJOCO_GL": "egl", "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True"})
)

# Create volume
volume = modal.Volume.from_name("tabrl-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # Upgrade to H100 for faster training!
    timeout=3600,  # 1 hour timeout
    volumes={"/models": volume},
)
def train_playground_locomotion(
    env_name: str = "Go1JoystickFlatTerrain",
    category: str = "locomotion",  # New parameter for category
    training_steps: int = 30_000_000,  # 30M steps as in notebook
    eval_episodes: int = 5,
    save_path: str = "/models",
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a policy using MuJoCo Playground environments.
    
    Args:
        env_name: Name of the environment
        category: Category (locomotion, manipulation, dm_control_suite)
        training_steps: Total training steps
        eval_episodes: Number of evaluation episodes
        save_path: Path to save the trained model
        run_id: Unique identifier for this training run
        
    Returns:
        Training results and model info
    """
    # Configure XLA for better performance
    os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Import everything inside the function for Modal
    import jax
    import jax.numpy as jp
    import numpy as np
    from datetime import datetime
    import functools
    import pickle
    import subprocess
    
    from mujoco_playground import registry
    from mujoco_playground import wrapper
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    
    print(f"🚀 Starting training for {category}/{env_name}")
    print(f"ppo.train type: {type(ppo.train)}, callable: {callable(ppo.train)}")
    
    # Check GPU availability
    try:
        gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("GPU Info:")
        print(gpu_info.stdout[:500])  # First 500 chars
    except:
        print("nvidia-smi not available")
    
    print(f"JAX devices: {jax.devices()}")
    
    # Load environment and config based on category
    registry_obj = getattr(registry, category)
    env = registry_obj.load(env_name)
    env_cfg = registry_obj.get_default_config(env_name)
    
    # Get PPO params - use env_cfg directly instead of locomotion_params
    if hasattr(env_cfg, 'training') and hasattr(env_cfg.training, 'ppo'):
        # Use config from environment
        ppo_params = env_cfg.training.ppo
        episode_length = env_cfg.episode_length if hasattr(env_cfg, 'episode_length') else 1000
    else:
        # Use default PPO config
        ppo_params = {
            'num_timesteps': training_steps,
            'num_evals': eval_episodes,
            'reward_scaling': 1,
            'episode_length': 1000,
            'normalize_observations': True,
            'action_repeat': 1,
            'unroll_length': 5,
            'num_minibatches': 32,
            'num_updates_per_batch': 4,
            'discounting': 0.97,
            'learning_rate': 3e-4,
            'entropy_cost': 1e-2,
            'num_envs': 512,
            'batch_size': 256,
        }
        episode_length = 1000
    
    print(f"Environment: {category}/{env_name}")
    print(f"Episode length: {episode_length}")
    print(f"Action dim: {env.action_size}")
    
    # Sample environment to check structure
    rng = jax.random.PRNGKey(0)
    sample_state = env.reset(rng)
    print(f"Observation keys: {list(sample_state.obs.keys())}")
    print(f"State obs shape: {sample_state.obs['state'].shape}")
    
    # Training progress tracking
    training_data = {
        'steps': [],
        'rewards': [],
        'episode_lengths': [],
        'times': []
    }
    start_time = datetime.now()
    
    def progress(num_steps, metrics):
        training_data['steps'].append(num_steps)
        training_data['rewards'].append(metrics['eval/episode_reward'])
        training_data['episode_lengths'].append(metrics['eval/episode_length'])
        training_data['times'].append(datetime.now())
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Step {num_steps:,}: "
              f"Reward: {metrics['eval/episode_reward']:.2f}, "
              f"Length: {metrics['eval/episode_length']:.1f}, "
              f"Time: {elapsed:.1f}s")
    
    # Setup training
    randomizer = None  # Other categories may not have domain randomizers
        
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    
    # Update training steps
    ppo_training_params['num_timesteps'] = training_steps
    
    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
        progress_fn=progress
    )
    
    print(f"🏋️ Training with {training_steps:,} steps...")
    
    # Run training
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry_obj.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Evaluate trained policy
    print("🧪 Evaluating trained policy...")
    eval_env = registry_obj.load(env_name, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    eval_rewards = []
    eval_lengths = []
    
    for episode in range(eval_episodes):
        rng = jax.random.PRNGKey(episode + 42)
        state = jit_reset(rng)
        episode_reward = 0.0
        episode_step_count = 0
        
        for step in range(episode_length):
            act_rng, rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            
            episode_reward += state.reward
            episode_step_count += 1
            
            if state.done:
                break
        
        eval_rewards.append(float(episode_reward))
        eval_lengths.append(episode_step_count)
        print(f"Eval episode {episode + 1}: {episode_reward:.2f} reward, {episode_step_count} steps")
    
    avg_reward = np.mean(eval_rewards)
    avg_length = np.mean(eval_lengths)
    print(f"📊 Evaluation: Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.1f}")
    
    # Save model
    if run_id is None:
        run_id = f"{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_filename = f"{run_id}_policy.pkl"
    model_path = f"{save_path}/{model_filename}"
    
    model_data = {
        'params': params,
        'make_inference_fn': make_inference_fn,
        'env_name': env_name,
        'env_cfg': env_cfg,
        'training_steps': training_steps,
        'training_time': training_time,
        'final_metrics': metrics,
        'eval_rewards': eval_rewards,
        'eval_lengths': eval_lengths,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'training_data': training_data,
        'obs_shape': sample_state.obs['state'].shape,
        'action_size': env.action_size,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved to {model_path}")
    
    return {
        'status': 'completed',
        'env_name': env_name,
        'training_steps': training_steps,
        'training_time': training_time,
        'model_path': model_path,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'eval_rewards': eval_rewards,
        'final_metrics': metrics,
    }

@app.function(image=image, volumes={"/models": volume})
def list_trained_models():
    """List all trained models"""
    import os
    models = []
    for file in os.listdir("/models"):
        if file.endswith(".pkl"):
            models.append(file)
    return sorted(models)

@app.function(image=image, volumes={"/models": volume})
def get_model_info(model_filename: str):
    """Get information about a trained model"""
    import pickle
    
    model_path = f"/models/{model_filename}"
    if not os.path.exists(model_path):
        return {"error": f"Model {model_filename} not found"}
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Return basic info (not the full model params which are large)
        return {
            'env_name': model_data['env_name'],
            'training_steps': model_data['training_steps'],
            'training_time': model_data['training_time'],
            'avg_reward': model_data['avg_reward'],
            'avg_length': model_data['avg_length'],
            'obs_shape': list(model_data['obs_shape']),
            'action_size': model_data['action_size'],
            'eval_rewards': model_data['eval_rewards'],
        }
    except Exception as e:
        return {"error": f"Error loading model: {str(e)}"}

if __name__ == "__main__":
    # Quick test
    print("Available environments:")
    from mujoco_playground import registry
    for env in sorted(registry.locomotion._envs.keys()):
        print(f"  - {env}")
