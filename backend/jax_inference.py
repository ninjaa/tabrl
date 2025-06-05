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
        
    def load_model(self, model_path: str) -> bool:
        """
        Load a JAX model from pickle file
        
        Args:
            model_path: Path to the .pkl model file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract the policy network parameters
            params = model_data.get('params', model_data.get('model_params'))
            if params is None:
                raise ValueError("No parameters found in model file")
                
            # Store model info
            model_id = Path(model_path).stem
            self.loaded_models[model_id] = {
                'params': params,
                'metadata': model_data.get('metadata', {}),
                'action_dim': model_data.get('action_dim', model_data.get('nu')),
                'observation_dim': model_data.get('observation_dim', model_data.get('nq_nv')),
            }
            
            print(f"Loaded model {model_id} successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict(self, model_id: str, observation: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference for a single observation
        
        Args:
            model_id: ID of the loaded model
            observation: Observation array
            
        Returns:
            Action array or None if error
        """
        if model_id not in self.loaded_models:
            print(f"Model {model_id} not loaded")
            return None
            
        try:
            model_info = self.loaded_models[model_id]
            params = model_info['params']
            
            # Convert numpy to JAX array
            obs_jax = jnp.array(observation)
            
            # Add batch dimension if needed
            if obs_jax.ndim == 1:
                obs_jax = obs_jax[None, :]
            
            # Run through the network
            # This assumes the standard Brax MLP policy structure
            # You might need to adjust based on actual network architecture
            action = self._forward_mlp(params, obs_jax)
            
            # Remove batch dimension and convert to numpy
            if action.shape[0] == 1:
                action = action[0]
                
            return np.array(action)
            
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
    
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
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a loaded model"""
        if model_id not in self.loaded_models:
            return None
            
        info = self.loaded_models[model_id]
        return {
            'model_id': model_id,
            'observation_dim': info['observation_dim'],
            'action_dim': info['action_dim'],
            'metadata': info['metadata']
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


if __name__ == "__main__":
    # Test inference
    test_obs = np.random.randn(24)  # Example observation
    print(f"Test observation shape: {test_obs.shape}")
    
    # Would load a real model here
    # inference_engine.load_model("path/to/model.pkl")
    # action = inference_engine.predict("model_id", test_obs)
    # print(f"Action: {action}")
