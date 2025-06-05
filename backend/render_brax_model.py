import pickle
import jax
import jax.numpy as jnp
from mujoco_playground import registry
from brax.io import html, image
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import acting
from brax.training import types
import imageio
from pathlib import Path
import argparse
import mujoco
import functools


def render_brax_model(model_path, output_prefix="rollout", episode_length=100):
    """Render video from Brax trained model with correct params structure."""
    
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    # Extract model info
    params = model_data['params']
    env_name = model_data.get('env_name', 'Go1JoystickFlatTerrain')
    obs_shape = model_data.get('obs_shape', (48,))
    action_size = model_data.get('action_size', 12)
    
    print(f"Model loaded successfully:")
    print(f"  Environment: {env_name}")
    print(f"  Training steps: {model_data.get('training_steps', 'N/A'):,}")
    print(f"  Avg reward: {model_data.get('avg_reward', 'N/A'):.2f}")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action size: {action_size}")
    
    # Load environment
    category = 'locomotion'  # Default, could be parameterized
    registry_obj = getattr(registry, category)
    env = registry_obj.load(env_name)
    eval_env = registry_obj.load(env_name)  # Separate eval env for rendering
    
    # Create network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        obs_shape[-1],
        action_size,
        preprocess_observations_fn=lambda x: x
    )
    
    # Create inference function manually
    # The params tuple contains: (normalizer_state, actor_params, critic_params)
    normalizer_state, actor_params, critic_params = params
    
    # Build the policy network
    ppo_network = network_factory()
    
    # Create the inference function
    def make_policy(params, deterministic=False):
        def policy(observations, key):
            # Extract state observation from dict if necessary
            if isinstance(observations, dict) and 'state' in observations:
                obs = observations['state']
            else:
                obs = observations
                
            # Normalize observations
            normalizer_state, actor_params, _ = params
            
            # Apply normalization if normalizer state exists
            if normalizer_state is not None:
                # Check if normalizer_state is a dict with mean/std
                if hasattr(normalizer_state, 'mean') and hasattr(normalizer_state, 'std'):
                    # Brax normalizer has mean/std as dicts with 'state' key
                    if isinstance(normalizer_state.mean, dict) and 'state' in normalizer_state.mean:
                        mean = normalizer_state.mean['state']
                        std = normalizer_state.std['state']
                    else:
                        mean = normalizer_state.mean
                        std = normalizer_state.std
                    
                    # Observations is just the state part, not a dict
                    normalized_obs = (obs - mean) / (std + 1e-8)
                else:
                    normalized_obs = obs
            else:
                normalized_obs = obs
            
            # Get actions from policy network
            # Use the params dict directly like in jax_inference.py
            network_params = actor_params['params']
            
            # Forward pass through network
            x = normalized_obs
            layer_idx = 0
            while f'hidden_{layer_idx}' in network_params:
                layer_name = f'hidden_{layer_idx}'
                w = network_params[layer_name]['kernel']
                b = network_params[layer_name]['bias']
                
                x = jnp.dot(x, w) + b
                
                # Check if this is the last layer
                if f'hidden_{layer_idx + 1}' not in network_params:
                    # Last layer - no activation for continuous actions
                    break
                else:
                    # Hidden layer - apply activation
                    x = jax.nn.swish(x)  # Brax uses swish activation
                
                layer_idx += 1
            
            actions = x
            
            # PPO outputs mean and log_std concatenated
            # Split to get just the mean for deterministic policy
            action_dim = actions.shape[-1] // 2
            action_mean = actions[..., :action_dim]
            # action_log_std = actions[..., action_dim:]  # Not needed for deterministic
            
            if deterministic:
                return action_mean, {}
            else:
                # For stochastic policy, would sample from Normal(action_mean, exp(action_log_std))
                return action_mean, {}
        
        return policy
    
    # Create jitted inference function
    inference_fn = make_policy(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Jit environment functions for speed
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    
    # Run rollout
    print(f"\nRunning rollout for {episode_length} steps...")
    rng = jax.random.PRNGKey(42)
    rollout = []
    
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    
    for i in range(episode_length):
        # Get action from policy
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, act_rng)
        
        # Step environment
        state = jit_step(state, action)
        rollout.append(state)
        
        if state.done:
            print(f"Episode done at step {i}")
            break
        
        if i % 20 == 0:
            print(f"  Step {i}/{episode_length}")
    
    print(f"Rollout complete with {len(rollout)} steps")
    
    # Render video using Brax's built-in rendering
    print("\nRendering video...")
    
    # Setup rendering options
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    
    # Render every nth frame for reasonable video length
    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj = rollout[::render_every]
    
    # Render frames
    frames = eval_env.render(
        traj, 
        camera="side", 
        scene_option=scene_option, 
        height=480, 
        width=640
    )
    
    # Save video
    video_path = Path(f"{output_prefix}.mp4")
    video_path.parent.mkdir(exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Saved video to {video_path}")
    
    print(f"\nRendering complete!")
    print(f"Video saved to: {video_path}")
    
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                       default="models/Go1JoystickFlatTerrain_reward_fn.pkl",
                       help="Path to the trained model")
    parser.add_argument("--output-prefix", type=str, default="models/rollout",
                       help="Prefix for output files")
    parser.add_argument("--episode-length", type=int, default=100,
                       help="Number of steps to simulate (100 = 2 seconds at 50Hz)")
    
    args = parser.parse_args()
    
    render_brax_model(
        model_path=args.model_path,
        output_prefix=args.output_prefix,
        episode_length=args.episode_length
    )
