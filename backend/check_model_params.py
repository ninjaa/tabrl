#!/usr/bin/env python3
"""Check the structure of parameters in the trained model"""

import modal

def check_model_structure():
    try:
        # Download the model
        download_fn = modal.Function.from_name("tabrl-playground-training", "download_model")
        print("Downloading model...")
        model_data = download_fn.remote("test_locomotion_Go1JoystickFlatTerrain_policy.pkl")
        
        if model_data is None:
            print("Model returned None")
            # Let's try to list models first
            list_fn = modal.Function.from_name("tabrl-playground-training", "list_trained_models")
            models = list_fn.remote()
            print("Available models:", models)
            return
        
        print("Model keys:", list(model_data.keys()))
        print("\nParams structure:")
        
        params = model_data.get('params', {})
        
        def print_param_structure(params, prefix=""):
            for key, value in params.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    print_param_structure(value, prefix + "  ")
                else:
                    # It's likely a JAX array
                    shape = getattr(value, 'shape', 'unknown')
                    dtype = getattr(value, 'dtype', 'unknown')
                    print(f"{prefix}{key}: shape={shape}, dtype={dtype}")
        
        print_param_structure(params)
        
        print(f"\nObs shape: {model_data.get('obs_shape')}")
        print(f"Action size: {model_data.get('action_size')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_structure()
