# Minimal training environment for testing

import modal
import numpy as np

app = modal.App("tabrl-training")

image = modal.Image.debian_slim().pip_install(
    "stable-baselines3[extra]",
    "gymnasium", 
    "mujoco",
    "onnx",
    "onnxruntime"
)

@app.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")  
def train_test_policy(request):
    """Test training with a simple policy"""
    
    # For now, just verify everything imports
    import stable_baselines3
    import gymnasium as gym
    import mujoco
    
    return {
        "status": "ready",
        "versions": {
            "sb3": stable_baselines3.__version__,
            "gym": gym.__version__,
            "mujoco": mujoco.__version__
        }
    }
