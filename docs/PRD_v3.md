# MuJoCo Browser Training Platform - PRD

## 🎯 Product Vision
Build a browser-based robotics training platform where users can teach robots new behaviors using natural language. Uses FastAPI backend for LLM integration and training, with MuJoCo WASM + ONNX inference in the browser.

## 📁 Current Architecture (Updated)

### Backend (FastAPI) 
- **Location**: `/backend/`
- **Key Files**:
  - `app.py` - FastAPI server with CORS for frontend
  - `inference.py` - LiteLLM integration for multi-provider LLM support
  - `training.py` - RL training engine with MuJoCo simulation
  - `scene_parser.py` - XML parsing and robot structure analysis
  - `rl_training.py` - PPO implementation with reward function compilation

### Frontend (React + Three.js)
- **Location**: `/frontend/`  
- **Key Files**:
  - `src/App.jsx` - Main React app with training pipeline UI
  - `src/App.css` - Modern styling with light/dark themes
  - Vite build system for fast development

### Scene Library Structure
```
/scenes/
├── manipulation/
│   ├── universal_robots_ur5e/
│   │   ├── scene.xml
│   │   ├── ur5e.xml (included robot)
│   │   └── meshes/...
├── locomotion/
│   ├── anybotics_anymal_c/
│   │   ├── scene.xml
│   │   └── meshes/...
├── humanoids/
│   ├── unitree_g1/
│   │   ├── scene.xml
│   │   └── meshes/...
└── hands/
    ├── shadow_hand/
        ├── scene.xml
        └── meshes/...
```

### MuJoCo WASM Integration (Future)
- **Location**: `/mujoco-wasm-build/` (planned)
- **Purpose**: Browser-based simulation and ONNX inference
- **Current Status**: Backend simulation, WASM integration planned

## 🏗️ Technical Architecture (Updated - Brax/JAX Integration)

### Core Components

1. **Frontend (React + Vite)**
   - Dynamic scene and model selection via API
   - Natural language task input
   - Reward function display and selection
   - Real-time training progress visualization
   - **MuJoCo WASM + ONNX.js inference** (browser-based policy execution)

2. **FastAPI Backend**
   - Multi-provider LLM integration (Claude, GPT-4, Gemini, DeepSeek)
   - **JAX/Brax training pipeline** (GPU-accelerated on Modal)
   - Scene parsing and robot structure analysis
   - **ONNX model export** from JAX/Flax policies
   - Progress tracking and status polling

3. **Training Pipeline (New Architecture)**
   - Scene-aware policy generation via LLM
   - Multiple reward approaches (dense, sparse, shaped)
   - **Brax PPO training** (7-minute locomotion policies)
   - **Modal GPU deployment** for distributed training
   - Automatic ONNX export for browser inference

### API Endpoints (Updated)
```
GET  /api/scenes                 - List available robot scenes
GET  /api/models/llm            - List available LLM models  
POST /api/model/select          - Switch LLM models
GET  /api/model/current         - Get current LLM model
POST /api/policy/generate       - Generate reward functions from natural language
POST /api/training/start        - Start Brax training job on Modal GPU
GET  /api/training/{id}/status  - Get training progress
GET  /api/models               - List trained ONNX models
GET  /api/models/{id}/export/onnx - Export trained policy to ONNX format
```

## 🚀 Current Status & Next Steps

### ✅ Completed (Phase 1-2)
- FastAPI backend with multi-LLM support
- Scene parsing and robot structure analysis
- Policy generation with proper reward function wrapping
- React frontend with dynamic data loading
- Training progress tracking and model export
- End-to-end pipeline validation

### 🔄 In Progress (Phase 3A - JAX/Brax Integration)
- JAX/Brax dependency integration
- Brax environment wrapper for existing MuJoCo scenes
- ONNX export pipeline development
- Local CPU testing and validation

### 📋 TODO (Phase 3B-C)
- Modal GPU training deployment
- MuJoCo WASM + ONNX.js frontend integration
- Training visualization (foot trajectories, velocity tracking)
- Policy comparison and analysis tools
- Public deployment and demo

## 📐 Training Architecture (Updated)

### Brax/JAX Training Stack
```python
# LLM generates reward function
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
    network_factory=ppo_networks.make_ppo_networks
)

# Export to ONNX for browser inference
export_to_onnx(params, obs_shape=(24,), action_shape=(12,))
```

### Modal GPU Deployment
```python
@modal.function(gpu="A100", timeout=600)  # 10-minute timeout
def train_locomotion_policy(scene_name, reward_code, config):
    # Load scene in Brax
    env = load_brax_environment(scene_name)
    
    # Inject LLM reward function
    env = env.replace(reward_fn=compile_reward(reward_code))
    
    # Train with Brax PPO
    inference_fn, params, metrics = ppo.train(env, **config)
    
    # Export to ONNX
    onnx_bytes = export_onnx(params, env.observation_size, env.action_size)
    
    return onnx_bytes, metrics
```

## 📋 User Flows (Current Implementation)

### Completed User Experience
1. **Landing Page**
   - Clean React interface with training pipeline tabs
   - API keys managed via .env file (developer setup)
   - Dynamic scene and model selection from backend
   
2. **Policy Generation Flow**
   ```
   User inputs: "Make the robot dance"
   Selects: Scene (e.g., Unitree G1 humanoid)
   Selects: LLM Model (Claude 4, GPT-4o, etc.)
   
   System generates:
   - Multiple reward functions (dense, sparse, shaped)
   - Proper function definitions with semantic APIs
   - Code preview in expandable cards
   ```

3. **Training Pipeline**
   ```
   User selects: Preferred reward approach
   Clicks: "Start Training"
   
   System shows:
   - Real-time training progress
   - Episode count and reward curves
   - ETA and status updates
   - Model export on completion
   ```

4. **Current Results**
   - Successful end-to-end training demonstrated
   - Models exported to ONNX format
   - Training videos and progress logs
   - 200+ reward achieved in demo tasks

### Next Phase: Browser Integration
1. **MuJoCo WASM Integration**
   - Replace backend simulation with browser WASM
   - ONNX inference directly in browser
   - 3D visualization with Three.js
   - No server dependency for inference

2. **Enhanced UI**
   - 3D scene viewer and manipulation
   - Monaco code editor for reward function editing
   - Real-time simulation preview
   - Model comparison and analysis

## 🚀 MVP Features (Hackathon Scope)

### Must Have
- [ ] Load pre-built scenes (3-5 examples)
- [ ] Natural language → Policy generation
- [ ] View generated code
- [ ] Start training with one click
- [ ] See training progress
- [ ] Run trained policy in browser
- [ ] Basic error handling

### Nice to Have
- [ ] Edit generated code
- [ ] Custom scene upload
- [ ] Save/load policies
- [ ] Share policies via URL
- [ ] Multiple policies per scene
- [ ] Training history

### Post-Hackathon
- [ ] Community policy library
- [ ] Advanced RL algorithms
- [ ] Multi-agent training
- [ ] Custom reward functions
- [ ] Policy composition

## 📊 Success Metrics

1. **Time to First Success**: < 5 minutes from landing to seeing a trained robot
2. **Training Success Rate**: > 70% of prompts result in working policies  
3. **Performance**: 60 FPS simulation, < 10ms inference latency
4. **Browser Compatibility**: Chrome, Firefox, Safari (latest versions)

## 🔧 Development Priorities

1. **Get Basic Flow Working**
   - Scene loads → User prompts → Training starts → Policy runs

2. **Polish the Experience**
   - Smooth animations
   - Clear status messages
   - Helpful error messages

3. **Add Power Features**
   - Code editing
   - Parameter tuning
   - Advanced scenes

## 📝 Example Prompts to Support

- "Pick up the [color] block"
- "Walk to the target"
- "Balance the pole"
- "Sort objects by size"
- "Avoid obstacles while moving forward"
- "Throw the ball into the basket"
- "Open the door"
- "Follow the moving target"

## 🐛 Error Handling

### Common Issues to Handle
1. **Invalid API Keys**: Clear message, re-prompt
2. **Training Timeout**: Offer to extend or download partial
3. **Code Generation Failure**: Retry with modified prompt
4. **Browser Compatibility**: Graceful degradation message
5. **WebContainer Boot Failure**: Fallback instructions

## 📚 Resources for Claude/Developers

### Key Libraries
- MuJoCo WASM: [github.com/google-deepmind/mujoco_wasm](https://github.com/google-deepmind/mujoco_wasm)
- StackBlitz WebContainers: [webcontainers.io](https://webcontainers.io)
- ONNX Runtime Web: [onnxruntime.ai/docs/get-started/with-javascript.html](https://onnxruntime.ai/docs/get-started/with-javascript.html)
- Modal: [modal.com/docs](https://modal.com/docs)

### File Structure to Create
```
/src/
├── components/
│   ├── SceneSelector.jsx
│   ├── MuJoCoViewer.jsx
│   ├── PromptInterface.jsx
│   ├── CodeEditor.jsx
│   ├── TrainingMonitor.jsx
│   └── PolicyRunner.jsx
├── webcontainer/
│   ├── server.py
│   ├── requirements.txt
│   └── inference.py
├── modal/
│   ├── generate_policy.py
│   └── train_policy.py
├── utils/
│   ├── mujoco_loader.js
│   ├── api_client.js
│   └── storage.js
└── App.jsx
```

## 🚀 Quick Start Guide for Development

### Initial Setup Actions (Day 1)
1. **Clone MuJoCo Resources**
   ```bash
   git clone https://github.com/google-deepmind/mujoco_menagerie
   git clone https://github.com/zalo/mujoco_wasm
   ```

2. **Set Up Modal**
   - Create Modal account
   - Install CLI: `pip install modal`
   - Deploy initial functions:
     ```bash
     modal deploy modal/generate_policy.py
     modal deploy modal/train_policy.py
     ```

3. **Frontend Viewer Integration**
   - Get viewer repo from co-hacker
   - Integrate with mujoco_wasm
   - Test basic scene loading

### First Implementation Tasks
1. **WebContainer Bootstrap**
   - Initialize StackBlitz WebContainers SDK
   - Create Flask server with settings API
   - Test key storage/retrieval
   - Verify proxy to Modal works

2. **Hello World Flow**
   ```javascript
   // Test each component:
   1. Load a simple scene (cartpole)
   2. Send test prompt to Modal
   3. Get generated policy back
   4. Start training job
   5. Download ONNX model
   6. Run inference in WebContainer
   ```

3. **Basic UI Components**
   - API key input form
   - Scene selector (just 3-4 scenes initially)
   - Simple prompt input
   - Status display

### Validation Checklist
- [ ] WebContainer boots successfully
- [ ] Settings persist in browser
- [ ] Modal functions callable via proxy
- [ ] ONNX model downloads work
- [ ] Inference runs at 60 FPS
- [ ] No CORS errors!

## 🎯 Definition of Done

The platform is complete when a user can:
1. Open the website
2. Select a robot scene  
3. Type what they want the robot to do
4. See the robot learning
5. Watch the trained robot perform the task
6. All within 5 minutes, in one browser tab

---

*This PRD should give Claude or any developer a complete picture of what to build. The key is emphasizing the browser-first approach and the magical experience of seeing a robot learn from natural language, all without leaving the webpage.*