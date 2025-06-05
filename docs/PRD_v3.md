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

## MuJoCo Playground Bridge Architecture

### The Critical Connection

**BREAKTHROUGH**: We can extract complete MuJoCo XML scenes from Playground environments and load them directly in the frontend MuJoCo WASM viewer.

```python
# Backend: Extract any Playground environment as XML
env = registry.locomotion.load("Go1JoystickFlatTerrain")
complete_xml = env.sys.to_xml()  # Full scene: robot + terrain + physics

# Frontend: Load directly in browser
mujoco.Model.load_from_xml(complete_xml)
# Result: User sees Go1 robot in complete outdoor park scene!
```

### Scene Export API

```python
@app.get("/api/playground/environments")
def list_playground_environments():
    """List all 52 available environments by category"""
    return {
        "locomotion": ["Go1JoystickFlatTerrain", "BerkeleyHumanoidJoystickFlatTerrain", ...],
        "manipulation": ["PandaPickCube", "AlohaHandOver", ...], 
        "dm_control_suite": ["HumanoidRun", "CheetahRun", ...]
    }

@app.get("/api/playground/{category}/{env_name}/xml")
def get_environment_xml(category: str, env_name: str):
    """Export complete MuJoCo XML for any playground environment"""
    registry_obj = getattr(registry, category)
    env = registry_obj.load(env_name)
    xml_string = env.sys.to_xml()
    
    return {
        "xml": xml_string,
        "env_name": env_name,
        "robot_type": extract_robot_type(xml_string),
        "scene_description": get_scene_description(env_name)
    }
```

### Frontend Scene Selector

```javascript
// 1. Populate scene dropdown
const environments = await fetch('/api/playground/environments').then(r => r.json());

// 2. User selects: "Go1JoystickFlatTerrain"
const selectedEnv = "Go1JoystickFlatTerrain";

// 3. Load complete scene XML
const {xml, robot_type} = await fetch(
    `/api/playground/locomotion/${selectedEnv}/xml`
).then(r => r.json());

// 4. Initialize MuJoCo with complete scene
const model = mujoco.Model.load_from_xml(xml);
const simulation = new mujoco.Simulation(model);

// 5. User now sees Go1 robot in full outdoor terrain!
```

### Scene Categories & Visual Contexts

| **Environment** | **Visual Context** | **Demo Appeal** |
|-----------------|-------------------|-----------------|
| `Go1JoystickFlatTerrain` | Outdoor park with grass terrain | High - recognizable robot |
| `BerkeleyHumanoidJoystickFlatTerrain` | Flat ground for human-like movement | High - emotional connection |
| `SpotJoystickGaitTracking` | Industrial/construction setting | Medium - professional appeal |
| `PandaPickCube` | Laboratory workspace with cube | Medium - precise manipulation |
| `AlohaHandOver` | Kitchen/workshop environment | High - relatable cooking tasks |

## JAX/Flax to ONNX Conversion Pipeline

### The Training → Browser Pipeline

```python
# 1. Train policy with Brax PPO (JAX/Flax)
policy_fn, params = train_brax_policy(env, reward_fn, steps=100_000)

# 2. Convert JAX model to ONNX for browser inference
def convert_jax_to_onnx(policy_fn, params, input_shape):
    # JAX → TensorFlow conversion
    import jax2tf
    tf_fn = jax2tf.convert(lambda x: policy_fn(params, x))
    
    # Create TensorFlow concrete function
    tf_concrete = tf_fn.get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    
    # TensorFlow → ONNX export
    import tf2onnx
    onnx_model, _ = tf2onnx.convert.from_function(
        tf_concrete,
        input_signature=[tf.TensorSpec(input_shape, tf.float32)]
    )
    
    return onnx_model

# 3. Save ONNX model to Modal volume
onnx_path = f"/models/{policy_id}.onnx"
save_onnx_model(onnx_model, onnx_path)

# 4. Return download URL for frontend
return {
    "policy_id": policy_id,
    "onnx_url": f"/api/models/{policy_id}.onnx",
    "input_shape": env.observation_space.shape,
    "output_shape": env.action_space.shape
}
```

### Browser Policy Execution

```javascript
// 1. Download trained ONNX model
const policyResponse = await fetch('/api/models/dance_policy_v1.onnx');
const policyBuffer = await policyResponse.arrayBuffer();

// 2. Load ONNX model in browser
const session = await ort.InferenceSession.create(policyBuffer);

// 3. Real-time policy inference loop
function simulationStep() {
    // Get current robot state from MuJoCo
    const observation = simulation.getObservation();
    
    // Run policy inference
    const feeds = {'input': new ort.Tensor('float32', observation, [1, obsSize])};
    const results = await session.run(feeds);
    const actions = results.output.data;
    
    // Apply actions to MuJoCo simulation
    simulation.setActions(actions);
    simulation.step();
    
    requestAnimationFrame(simulationStep);
}
```

### Performance Optimization

**Training Time Targets:**
- **Command-based**: 0 seconds (direct velocity command generation)
- **Fine-tuning**: 30-60 seconds (build on pre-trained locomotion) 
- **Full training**: 8-15 minutes (complete custom behavior)

**Model Size Optimization:**
- **Policy compression**: Quantize ONNX to FP16 for faster download
- **Batch conversion**: Pre-convert popular base policies
- **Progressive loading**: Start with low-quality, upgrade to high-quality

**Browser Performance:**
- **ONNX.js optimization**: Use WebGL/WASM backends
- **Inference caching**: Cache policy outputs for repeated states
- **Frame rate targets**: 30 FPS simulation with real-time policy execution

## Intelligent Policy Generation System

TabRL features an **intelligent policy generation API** that automatically selects the optimal training approach based on task complexity, time budget, and quality requirements.

### Auto-Strategy Selection

The system tries approaches from **fastest to slowest**:

#### **Level 1: Command-Based Generation (0 seconds)**
- Generate velocity commands directly from task description
- Use existing pre-trained joystick policies 
- Best for: Dance, choreography, simple behaviors
- Success rate: ~70% for motion tasks

#### **Level 2: Fine-Tuning (30-60 seconds)**  
- Quick fine-tune existing policies with custom rewards
- Leverage pre-trained locomotion/manipulation skills
- Best for: Task-specific behaviors, style variations
- Success rate: ~90% for similar tasks

#### **Level 3: Full Training (8+ minutes)**
- Train from scratch with completely custom rewards
- Maximum flexibility and task coverage
- Best for: Novel tasks, complex multi-objective behaviors
- Success rate: ~95% for any feasible task

### Smart API Design

```python
POST /api/generate-policy
{
  "task_description": "Make robot dance to electronic music",
  "robot": "Go1", 
  "time_budget": "fast|medium|unlimited",
  "quality_threshold": 0.8,
  "scene_context": "urban_playground"
}

Response:
{
  "policy_id": "dance_policy_v1",
  "method_used": "fine_tuned", 
  "training_time": "45 seconds",
  "quality_score": 0.87,
  "alternatives_tried": ["command_based"],
  "model_ready": true,
  "onnx_url": "/models/dance_policy_v1.onnx"
}
```

### Quality Scoring System

Each approach gets evaluated on:
- **Task completion accuracy** (0-1)
- **Movement naturalness** (0-1) 
- **Safety/stability** (0-1)
- **Energy efficiency** (0-1)

## Recommended Robot Selection

Based on MuJoCo Playground's 52 available environments, we recommend these **hero robots** for maximum visual impact and user engagement:

### Locomotion Stars (Primary Focus)

#### **Boston Dynamics Go1 Quadruped**
- **Environments**: `Go1JoystickFlatTerrain`, `Go1JoystickRoughTerrain`
- **Why**: Most recognizable robot, great for outdoor scenes
- **Best demos**: Parkour, dancing, following, patrol

#### **Berkeley Humanoid** 
- **Environments**: `BerkeleyHumanoidJoystickFlatTerrain`, `BerkeleyHumanoidJoystickRoughTerrain`
- **Why**: Most human-like, emotional connection
- **Best demos**: Dancing, martial arts, sports, social interaction

#### **Boston Dynamics Spot**
- **Environments**: `SpotJoystickGaitTracking`, `SpotFlatTerrainJoystick`
- **Why**: Industrial icon, professional appeal
- **Best demos**: Construction sites, inspection, security patrol

### Manipulation Masters (Secondary)

#### **Aloha Bimanual Robot**
- **Environments**: `AlohaHandOver`, `AlohaSinglePegInsertion`
- **Why**: Dual-arm coordination, impressive dexterity
- **Best demos**: Cooking, assembly, collaborative tasks

#### **Franka Panda Arm**
- **Environments**: `PandaPickCube`, `PandaOpenCabinet`, `PandaRobotiqPushCube`
- **Why**: Industry standard, precise manipulation
- **Best demos**: Laboratory work, delicate assembly, bartending

### Classic Control (Tertiary)

#### **Humanoid Walker**
- **Environments**: `HumanoidRun`, `HumanoidWalk`, `HumanoidStand`
- **Why**: Athletic movements, sports scenarios
- **Best demos**: Running, gymnastics, rehabilitation

### Scene Pairing Recommendations

| Robot | Ideal Scene | Demo Scenario |
|-------|-------------|---------------|
| Go1 | Urban park, warehouse | Package delivery, security patrol |
| Berkeley Humanoid | Living room, dance studio | Home assistant, entertainment |
| Spot | Construction site, factory | Industrial inspection, maintenance |
| Aloha | Kitchen, workshop | Cooking demo, assembly line |
| Panda | Laboratory, bar | Scientific experiments, cocktail making |

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