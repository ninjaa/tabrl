"""Export trained Brax models to ONNX format"""
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from typing import Tuple
import onnx
import tf2onnx
import tensorflow as tf
from mujoco_playground import registry, wrapper
from brax.training.agents.ppo import networks as ppo_networks


def jax_to_tf_function(jax_fn, input_spec):
    """Convert JAX function to TensorFlow function"""
    @tf.function
    def tf_fn(x):
        # Convert TF tensor to numpy, then JAX
        x_np = x.numpy()
        x_jax = jnp.array(x_np)
        # Run JAX function
        output_jax = jax_fn(x_jax)
        # Convert back to TF
        return tf.constant(np.array(output_jax))
    
    # Trace the function
    concrete_fn = tf_fn.get_concrete_function(input_spec)
    return concrete_fn


def export_brax_to_onnx(
    model_path: str,
    output_path: str,
    env_name: str = "Go1JoystickFlatTerrain",
    category: str = "locomotion"
) -> str:
    """Export a trained Brax model to ONNX format
    
    Args:
        model_path: Path to saved .pkl model
        output_path: Path for output .onnx file
        env_name: Environment name
        category: Environment category
        
    Returns:
        Path to saved ONNX file
    """
    # Load saved model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    params = model_data['params']
    obs_shape = model_data['obs_shape']
    action_size = model_data['action_size']
    
    # Recreate environment to get network factory
    registry_obj = getattr(registry, category)
    env = registry_obj.load(env_name)
    env_cfg = registry_obj.get_default_config(env_name)
    
    # Create network
    ppo_network = ppo_networks.make_ppo_networks(
        obs_shape[0],
        action_size,
        preprocess_observations_fn=lambda x: x  # No preprocessing for ONNX
    )
    
    # Create inference function
    def policy_fn(obs):
        """Pure policy function for inference"""
        policy_params = params[0] if isinstance(params, tuple) else params
        logits, _ = ppo_network.policy_network.apply(policy_params, obs)
        return logits
    
    # JIT compile
    jit_policy = jax.jit(policy_fn)
    
    # Create sample input
    sample_obs = jnp.zeros((1, obs_shape[0]), dtype=jnp.float32)
    
    # Option 1: Direct JAX to ONNX (if jax2onnx is available)
    try:
        import jax2onnx
        onnx_model = jax2onnx.convert(jit_policy, sample_obs)
        onnx.save(onnx_model, output_path)
        print(f"✅ Exported to ONNX using jax2onnx: {output_path}")
        return output_path
    except ImportError:
        print("jax2onnx not available, using TensorFlow bridge...")
    
    # Option 2: JAX -> TensorFlow -> ONNX
    # Create TF input spec
    input_spec = tf.TensorSpec(shape=(None, obs_shape[0]), dtype=tf.float32)
    
    # Convert to TF function
    tf_fn = jax_to_tf_function(jit_policy, input_spec)
    
    # Convert TF to ONNX
    onnx_model, _ = tf2onnx.convert.from_function(
        tf_fn,
        input_signature=[input_spec],
        opset=13,
        output_path=output_path
    )
    
    print(f"✅ Exported to ONNX via TensorFlow: {output_path}")
    return output_path


def main():
    """Example usage"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .pkl model')
    parser.add_argument('--output', required=True, help='Output .onnx path')
    parser.add_argument('--env', default='Go1JoystickFlatTerrain')
    parser.add_argument('--category', default='locomotion')
    args = parser.parse_args()
    
    export_brax_to_onnx(
        args.model,
        args.output,
        args.env,
        args.category
    )


if __name__ == "__main__":
    main()
