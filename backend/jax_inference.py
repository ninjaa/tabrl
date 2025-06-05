"""
Server-side JAX inference for trained policies
Since we can't easily export to ONNX due to dependency conflicts,
we'll run inference on the backend for the demo
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional
import pickle
from pathlib import Path


class JAXPolicyInference:
    """Run JAX policy inference on the backend"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        
    def load_model_from_modal(self, model_name: str) -> bool:
        """
        Load a JAX model from Modal volume
        
        Args:
            model_name: Name of the model file in Modal volume
            
        Returns:
            True if loaded successfully
        """
        try:
            # Try to download from Modal
            from modal import Function
            download_fn = Function.from_name("tabrl-playground-training", "download_model")
            model_data = download_fn.remote(model_name)
            
            if model_data is None:
                print(f"Model {model_name} not found in Modal volume")
                return False
            
            # Extract the policy network parameters
            params = model_data.get('params', model_data.get('model_params'))
            if params is None:
                raise ValueError("No parameters found in model file")
                
            # Extract dimensions from Playground training format
            obs_shape = model_data.get('obs_shape', [])
            obs_dim = obs_shape[0] if obs_shape else model_data.get('observation_dim', model_data.get('nq_nv'))
            action_dim = model_data.get('action_size', model_data.get('action_dim', model_data.get('nu')))
            
            # Store model info
            model_id = model_name.replace('.pkl', '')
            self.loaded_models[model_id] = {
                'params': params,
                'metadata': {
                    'env_name': model_data.get('env_name', ''),
                    'training_steps': model_data.get('training_steps', 0),
                    'avg_reward': float(model_data.get('avg_reward', 0)),
                    'training_time': model_data.get('training_time', 0),
                },
                'action_dim': action_dim,
                'observation_dim': obs_dim,
                'obs_shape': obs_shape,
            }
            
            print(f"Loaded model {model_id} from Modal successfully")
            print(f"  - Observation dim: {obs_dim}")
            print(f"  - Action dim: {action_dim}")
            print(f"  - Avg reward: {self.loaded_models[model_id]['metadata']['avg_reward']:.2f}")
            return True
            
        except Exception as e:
            print(f"Failed to load model from Modal: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def load_model(self, model_path: str) -> bool:
        """
        Load a JAX model from pickle file or Modal volume
        
        Args:
            model_path: Path to the .pkl model file or model name in Modal
            
        Returns:
            True if loaded successfully
        """
        # Check if it's a Modal model name (just filename, no path)
        if '/' not in model_path and model_path.endswith('.pkl'):
            return self.load_model_from_modal(model_path)
            
        # Otherwise try to load from local file
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract the policy network parameters
            params = model_data.get('params', model_data.get('model_params'))
            if params is None:
                raise ValueError("No parameters found in model file")
                
            # Extract dimensions 
            obs_shape = model_data.get('obs_shape', [])
            obs_dim = obs_shape[0] if obs_shape else model_data.get('observation_dim', model_data.get('nq_nv'))
            action_dim = model_data.get('action_size', model_data.get('action_dim', model_data.get('nu')))
            
            # Store model info
            model_id = Path(model_path).stem
            self.loaded_models[model_id] = {
                'params': params,
                'metadata': model_data.get('metadata', {}),
                'action_dim': action_dim,
                'observation_dim': obs_dim,
                'obs_shape': obs_shape,
            }
            
            print(f"Loaded model {model_id} successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict(self, model_id: str, observation: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference with loaded JAX model
        """
        if model_id not in self.loaded_models:
            print(f"Model {model_id} not loaded")
            return None
            
        try:
            model_data = self.loaded_models[model_id]
            params = model_data['params']
            
            # Convert observation to JAX array
            obs_jax = jnp.array(observation, dtype=jnp.float32)
            
            # Handle Brax PPO model structure
            if isinstance(params, tuple) and len(params) == 3:
                # params = (normalizer_state, actor_params, critic_params)
                normalizer_state, actor_params, critic_params = params
                
                # Normalize observation if normalizer exists
                if hasattr(normalizer_state, 'mean') and hasattr(normalizer_state, 'std'):
                    # Brax normalizer has mean/std as dicts with 'state' and 'privileged_state' keys
                    if isinstance(normalizer_state.mean, dict) and 'state' in normalizer_state.mean:
                        mean = normalizer_state.mean['state']
                        std = normalizer_state.std['state']
                    else:
                        mean = normalizer_state.mean
                        std = normalizer_state.std
                    
                    obs_normalized = (obs_jax - mean) / (std + 1e-8)
                else:
                    obs_normalized = obs_jax
                
                # Ensure batch dimension
                if obs_normalized.ndim == 1:
                    obs_normalized = obs_normalized[None, :]
                
                # Run through actor network
                action = self._forward_brax_network(actor_params['params'], obs_normalized)
                
                # PPO outputs mean and log_std concatenated, we only need mean for deterministic inference
                # If output dim is 2x expected action dim, take first half
                if action.shape[-1] == 2 * model_data.get('action_dim', action.shape[-1] // 2):
                    action = action[..., :action.shape[-1] // 2]
                
                # Remove batch dimension
                if action.shape[0] == 1:
                    action = action[0]
            else:
                # Legacy format or different structure
                if obs_jax.ndim == 1:
                    obs_jax = obs_jax[None, :]
                
                action = self._forward_mlp(params, obs_jax)
                
                if action.shape[0] == 1:
                    action = action[0]
                    
            return np.array(action)
            
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _forward_brax_network(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through Brax PPO network
        """
        # Brax uses naming like 'hidden_0', 'hidden_1', etc.
        # The last layer directly outputs actions without an activation
        
        layer_idx = 0
        while f'hidden_{layer_idx}' in params:
            layer_name = f'hidden_{layer_idx}'
            w = params[layer_name]['kernel']
            b = params[layer_name]['bias']
            
            x = jnp.dot(x, w) + b
            
            # Check if this is the last layer
            if f'hidden_{layer_idx + 1}' not in params:
                # Last layer - no activation for continuous actions
                break
            else:
                # Hidden layer - apply activation
                x = jax.nn.swish(x)  # Brax typically uses swish activation
            
            layer_idx += 1
            
        return x
    
    def _forward_mlp(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through MLP policy network
        Assumes standard Brax policy structure
        """
        # This is a simplified version - actual structure depends on Brax version
        # and network configuration used during training
        
        # Example for a 2-layer MLP:
        # Layer 1
        if 'policy/~/linear_0' in params:
            # Flax style parameters
            w1 = params['policy/~/linear_0']['kernel']
            b1 = params['policy/~/linear_0']['bias']
            x = jnp.tanh(jnp.dot(x, w1) + b1)
            
            # Layer 2
            w2 = params['policy/~/linear_1']['kernel']
            b2 = params['policy/~/linear_1']['bias']
            x = jnp.tanh(jnp.dot(x, w2) + b2)
            
            # Output layer
            w_out = params['policy/~/linear_2']['kernel']
            b_out = params['policy/~/linear_2']['bias']
            x = jnp.dot(x, w_out) + b_out
            
        else:
            # Try older param style
            x = jnp.tanh(jnp.dot(x, params['hidden_0']['w']) + params['hidden_0']['b'])
            x = jnp.tanh(jnp.dot(x, params['hidden_1']['w']) + params['hidden_1']['b'])
            x = jnp.dot(x, params['output']['w']) + params['output']['b']
        
        return x
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a loaded model"""
        if model_id not in self.loaded_models:
            return {}
        
        model = self.loaded_models[model_id]
        return {
            'model_id': model_id,
            'observation_dim': model.get('observation_dim'),
            'action_dim': model.get('action_dim'),
            'metadata': model.get('metadata', {}),
            'obs_shape': model.get('obs_shape', [])
        }


# Global inference engine instance
inference_engine = JAXPolicyInference()


def run_inference_server(model_path: str, observation: list) -> Optional[list]:
    """
    Simple function for API endpoint
    
    Args:
        model_path: Path to .pkl model file
        observation: List of observation values
        
    Returns:
        List of action values or None
    """
    model_id = Path(model_path).stem
    
    # Load model if not already loaded
    if model_id not in inference_engine.loaded_models:
        success = inference_engine.load_model(model_path)
        if not success:
            return None
    
    # Run inference
    action = inference_engine.predict(model_id, np.array(observation))
    
    if action is not None:
        return action.tolist()
    
    return None


# Example WebSocket handler for real-time inference
async def handle_inference_websocket(websocket, path):
    """
    WebSocket handler for streaming inference
    Expects JSON messages with:
    {
        "model_id": "model_name",
        "observation": [0.1, 0.2, ...]
    }
    """
    import json
    
    async for message in websocket:
        try:
            data = json.loads(message)
            model_id = data['model_id']
            observation = np.array(data['observation'])
            
            # Run inference
            action = inference_engine.predict(model_id, observation)
            
            if action is not None:
                response = {
                    'action': action.tolist(),
                    'status': 'success'
                }
            else:
                response = {
                    'action': None,
                    'status': 'error',
                    'message': 'Inference failed'
                }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            error_response = {
                'action': None,
                'status': 'error',
                'message': str(e)
            }
            await websocket.send(json.dumps(error_response))


# Create global instance for use in app.py
inference_engine = JAXPolicyInference()


if __name__ == "__main__":
    # Test inference
    test_obs = np.random.randn(24)  # Example observation
    print(f"Test observation shape: {test_obs.shape}")
    
    # Would load a real model here
    # inference_engine.load_model("path/to/model.pkl")
    # action = inference_engine.predict("model_id", test_obs)
    # print(f"Action: {action}")
