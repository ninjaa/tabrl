# MuJoCo Browser Training Platform - PRD

## 🎯 Product Vision
Build a browser-based robotics training platform where users can teach robots new behaviors using natural language. Uses FastAPI backend for LLM integration and training, with MuJoCo WASM simulation and server-side JAX inference.

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Frontend (React)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Scene     │  │   Training   │  │    MuJoCo WASM      │    │
│  │  Selector   │  │   Progress   │  │    Simulation       │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ API Calls + WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │     LLM     │  │   Training   │  │   JAX Inference     │    │
│  │ Integration │  │   Engine     │  │    (Server-side)    │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ GPU Training
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Modal (Serverless GPU)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Brax/JAX    │  │   Model      │  │    Training         │    │
│  │  Training   │  │   Storage    │  │    Container        │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Stack

### Backend (FastAPI)
- **Purpose**: LLM integration, training orchestration, model inference
- **Key Features**:
   - Multi-provider LLM support (Claude, GPT-4, Gemini, DeepSeek)
   - Reward function generation from natural language
   - Scene management and MuJoCo XML parsing
   - WebSocket streaming for real-time JAX inference
   - Modal integration for GPU training

### Frontend (React + Vite)
- **Purpose**: Browser-based simulation and control interface
- **Key Features**:
   - Scene selection and visualization
   - Natural language task input
   - Training progress monitoring
   - **MuJoCo WASM simulation** (robot visualization)
   - WebSocket client for real-time inference

### Training (Modal + Brax)
- **Purpose**: GPU-accelerated RL training
- **Key Features**:
   - Brax PPO training with JAX acceleration
   - **JAX/Flax policy models**
   - GPU containers (H100 for speed)
   - Model storage in Modal volumes
   - 7-minute target for locomotion policies
   - Server-side JAX inference (dependency compatibility)

## 📝 API Endpoints

```
POST /api/policy/generate      - Generate reward functions from prompts
POST /api/training/start       - Start GPU training with reward code
GET  /api/training/{id}/status - Get training progress
GET  /api/playground/scenes    - List available robot scenes
GET  /api/models               - List trained JAX models
GET  /api/models/{id}/inference - WebSocket endpoint for JAX inference
POST /api/llm/select           - Switch between LLM providers
```

## 🚀 Implementation Phases

### Phase 1: Core Platform ✅
- FastAPI backend with LLM integration
- Basic frontend with scene selection
- Modal GPU training setup
- MuJoCo Playground integration

### Phase 2: Training Pipeline ✅
- Brax/JAX training implementation
- Modal container deployment
- JAX model storage and retrieval
- Basic inference endpoints

### Phase 3: Browser Integration 🚧
- MuJoCo WASM scene rendering
- WebSocket streaming for inference
- Real-time policy execution
- Training visualization

### Phase 4: Polish & Scale
- Pre-trained model library
- Multi-robot support
- Advanced reward shaping
- Performance optimization

## 🧠 Smart Policy Generation

Generate custom robot behaviors using Brax PPO training with LLM-generated rewards:

```python
# LLM generates semantic reward function
prompt = "Make the robot do a backflip"
reward_code = llm.generate_reward(prompt)

# Example generated reward:
def locomotion_reward(state, action):
    base_pos = get_body_position(state, 'torso')
    return base_pos[0] * 2.0  # Forward progress

# Brax environment with custom reward
env = create_brax_env(
    scene_xml='locomotion/anymal_c/scene.xml',
    reward_fn=locomotion_reward,
    domain_randomization=llm_generated_config
)

# GPU training on Modal (7 minutes for locomotion)
make_inference_fn, params, metrics = ppo.train(
    environment=env,
    num_timesteps=30_000_000,
    num_evals=10
)

# Server-side JAX inference via WebSocket
async def inference_loop(websocket, params):
    while True:
        obs = await websocket.receive()
        action = inference_fn(params, obs)
        await websocket.send(action)
```

### Modal Function Signature
```python
@modal.function(gpu="H100", timeout=600)  # 10-minute timeout
def train_locomotion_policy(scene_name, reward_code, config):
    # Load scene in Brax
    env = load_brax_environment(scene_name)
    
    # Inject LLM reward function
    env = env.replace(reward_fn=compile_reward(reward_code))
    
    # Train with Brax PPO
    inference_fn, params, metrics = ppo.train(env, **config)
    
    # Save JAX model params
    model_path = save_jax_params(params, env.observation_size, env.action_size)
    
    return model_path, metrics
```

## 📋 User Flows (Current Implementation)

### 1. Task Definition
{{ ... }}

### Smart Policy Selection System

The platform intelligently chooses between three approaches based on task complexity:

1. **Command-based (0 seconds)**: Direct velocity commands
   - Success rate: ~70% for basic motions
   - Use case: Simple choreography, dance moves

2. **Fine-tuning (30-60 seconds)**: Quick adaptation of pre-trained policies
   - Success rate: ~90% for similar tasks  
   - Use case: Task variations, style transfer

3. **Full training (8+ minutes)**: Complete policy training from scratch
   - Success rate: ~95% for any feasible task
   - Use case: Novel behaviors, complex objectives

## 🏗️ System Components

### LLM-Generated Reward Functions
{{ ... }}

### Browser Simulation Stack
- **MuJoCo WASM**: Physics simulation in browser
- **WebSocket streaming**: Real-time JAX inference from server
- **React components**: Training UI and robot visualization

## 🔄 Training Pipeline

1. **Reward Generation**: LLM creates reward function from natural language
2. **GPU Training**: Modal container runs Brax PPO training
3. **Model Storage**: Trained JAX params saved to Modal volume
4. **Inference Ready**: Server loads model for WebSocket streaming

### Progressive Training Strategy

```python
async def generate_policy_progressive(task, robot, time_budget):
    # Try approaches from fastest to slowest
    if time_budget == "fast":
        # Level 1: Command-based
        return generate_command_policy(task)
    
    elif time_budget == "medium":
        # Level 2: Fine-tuning
        base_policy = load_pretrained(robot)
        reward_fn = generate_reward(task)
        return finetune_policy(base_policy, reward_fn, steps=1M)
    
    else:  # unlimited
        # Level 3: Full training
        reward_fn = generate_reward(task)
        return train_from_scratch(reward_fn, steps=30M)
```

## 🔑 Key Features

### Multi-Provider LLM Support
{{ ... }}

### Brax/JAX Training Stack
{{ ... }}

### JAX/Flax Model Pipeline

```python
# 1. Train policy with Brax PPO
from brax.training.agents.ppo import train as ppo
make_inference_fn, params, metrics = ppo.train(env, config)

# 2. Save JAX params for server-side inference
def save_jax_params(params, obs_size, action_size):
    model_data = {
        'params': params,
        'obs_size': obs_size,
        'action_size': action_size,
        'timestamp': datetime.now()
    }
    
    path = f"/models/{policy_id}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return path

# 3. Load and run inference on server
def load_jax_model(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Recreate inference function
    policy_network = make_ppo_networks(
        model_data['obs_size'],
        model_data['action_size']
    )
    
    return policy_network, model_data['params']
```

### Frontend Integration

```javascript
// WebSocket connection for real-time inference
const ws = new WebSocket('ws://localhost:8000/api/models/my_policy/inference');

// MuJoCo simulation loop
function simulationStep() {
    // Get robot state from MuJoCo
    const observation = mujoco.getState();
    
    // Send to server for JAX inference
    ws.send(JSON.stringify({observation}));
    
    // Receive action from server
    ws.onmessage = (event) => {
        const action = JSON.parse(event.data);
        mujoco.setControl(action);
    };
}
```

## 🎯 Success Metrics

- **Training time**: < 60 seconds for fine-tuning, < 8 minutes for full training
- **Success rate**: 90%+ task completion
- **Latency**: < 50ms inference round-trip
- **Scale**: Support 100+ concurrent training jobs

## 📊 Performance Optimization

- **GPU utilization**: H100 for fastest training
- **JAX compilation**: JIT-compiled inference functions
- **WebSocket efficiency**: Binary protocol for observations/actions
- **Caching**: Pre-trained base policies for common robots

## 🔄 API Response Examples

### Policy Generation Response
```json
{
  "task_description": "Make robot dance to electronic music",
  "policy_id": "dance_policy_v1",
  "training_method": "fine_tuning",
  "base_policy": "locomotion_base_v2",
  "training_time_estimate": "45 seconds",
  "reward_functions": [
    {
      "name": "dance_reward",
      "type": "dense",
      "code": "def compute_reward(state, action):\n    ..."
    }
  ]
}
```

{{ ... }}

## 🏃 Getting Started

{{ ... }}

## 📚 References

- MuJoCo Documentation: [mujoco.org](https://mujoco.org)
- Brax Physics Engine: [github.com/google/brax](https://github.com/google/brax)
- JAX Documentation: [jax.readthedocs.io](https://jax.readthedocs.io)
- Modal Platform: [modal.com/docs](https://modal.com/docs)
- MuJoCo WASM: [github.com/google-deepmind/mujoco_wasm](https://github.com/google-deepmind/mujoco_wasm)

## 🧪 Testing Checklist

### Backend Tests
{{ ... }}

### Training Pipeline Tests
- [ ] Modal GPU training launches successfully
- [ ] Brax PPO converges for test tasks
- [ ] JAX models save and load correctly
- [ ] WebSocket inference streams smoothly

### Frontend Tests
- [ ] MuJoCo WASM loads robot scenes
- [ ] WebSocket connection maintains stability
- [ ] Training progress updates in real-time
- [ ] Robot responds to inferred actions

### Integration Tests
- [ ] End-to-end: prompt → training → inference → robot motion
- [ ] Multiple concurrent training jobs work
- [ ] Server handles inference for multiple clients
- [ ] System recovers from training failures