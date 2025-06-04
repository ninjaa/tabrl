# MuJoCo Browser Training Platform - PRD

## 🎯 Product Vision
Build a browser-based robotics training platform where users can teach robots new behaviors using natural language, with all computation happening in a single browser tab (except LLM calls and GPU training).

## 📁 Existing Assets

### MuJoCo WASM Setup
- **Location**: `/mujoco-wasm-build/`
- **Key Files**:
  - `mujoco.js` - Main WASM wrapper
  - `mujoco.wasm` - Compiled MuJoCo binary
  - `mujoco_renderer.js` - WebGL rendering integration
  
### Scene Library Structure
```
/scenes/
├── robots/
│   ├── arm_6dof/
│   │   ├── robot.xml          # MuJoCo scene definition
│   │   ├── meshes/
│   │   │   ├── base.obj
│   │   │   ├── base.mtl
│   │   │   ├── link1.obj
│   │   │   └── ...
│   │   └── thumbnail.png
│   ├── humanoid_simple/
│   │   ├── robot.xml
│   │   ├── meshes/...
│   │   └── thumbnail.png
│   └── quadruped/
│       └── ...
├── environments/
│   ├── table_blocks/
│   │   ├── env.xml
│   │   ├── objects/...
│   │   └── thumbnail.png
│   └── obstacle_course/
│       └── ...
└── manifest.json              # Scene metadata
```

### API Keys Required
- **Claude/Anthropic API**: For LLM policy generation
- **Modal API**: For GPU training infrastructure
- **User provides these on first use**

## 🏗️ Technical Architecture

### Core Components

1. **Frontend (React/Vue)**
   - Scene selector UI
   - 3D viewport (Three.js + MuJoCo WASM)
   - Natural language prompt interface
   - Code editor (Monaco)
   - Training progress visualization
   - API key management

2. **WebContainer (StackBlitz SDK)**
   - Python environment (Pyodide)
   - Flask/FastAPI server for API proxying
   - ONNX Runtime for inference
   - Policy management

3. **Modal Functions**
   - LLM policy generation endpoint
   - RL training with GPU support
   - Model export to ONNX format

## 📋 User Flows

### First-Time User
1. **Landing Page**
   - Enter API keys (stored in localStorage)
   - Choose demo scene or upload custom
   
2. **Demo Experience**
   ```
   User sees: Robot arm with colored blocks
   Prompt: "Pick up the red block"
   System: Shows LLM generating code → Training progress → Robot learning
   Result: Robot successfully picks up red block
   ```

3. **Training New Behavior**
   ```
   User types: "Stack the blocks by size"
   System shows:
   - LLM generating policy code
   - Code preview (editable)
   - Hyperparameter settings
   - "Start Training" button
   - Real-time reward curve
   - ETA: ~3 minutes
   ```

4. **Testing Trained Policy**
   - Download completes (~10MB ONNX)
   - Auto-loads in WebContainer
   - Robot performs learned behavior
   - User can save/share policy

### Returning User
- Saved policies in localStorage
- Recent scenes remembered
- Can immediately load and run previous work

## 📐 Policy Template System

### Pre-Built Templates (Ship with TabRL)
Templates are pre-defined, tested policy architectures for different robot types. The LLM selects and customizes these rather than generating from scratch.

```python
# /templates/policies/
├── manipulator_policy.py    # 6-7 DOF arms with grippers
├── locomotion_policy.py      # Bipeds, quadrupeds, humanoids  
├── simple_control_policy.py  # Cartpole, reacher, etc.
└── base_template.py          # Shared utilities
```

Each template includes:
- Proven network architecture for that morphology
- Appropriate hyperparameters
- Stable Baselines3 compatible format
- ONNX export compatibility

## 🔌 Backend API Design

### Modal Endpoints

#### 1. Policy Generation
```python
POST /api/generate_policy
{
    "prompt": "Pick up the red block",
    "scene_xml": "<mujoco>...</mujoco>",
    "scene_name": "panda_tabletop"
}

Response:
{
    "policy_code": "class PolicyNetwork...",
    "template_used": "manipulator",
    "explanation": "Using manipulation template with grip action",
    "hyperparameters": {...},
    "estimated_training_time": "3 minutes"
}
```

#### 2. Training Management
```python
POST /api/train_policy
{
    "policy_code": "...",
    "scene_xml": "...",
    "hyperparameters": {...},
    "webhook_url": "wss://..."  # For progress updates
}

Response:
{
    "job_id": "train_abc123",
    "status": "started",
    "estimated_completion": "2024-01-15T10:30:00Z"
}

GET /api/training_status/{job_id}
Response:
{
    "status": "training",
    "progress": 0.45,
    "current_reward": 125.3,
    "eta_seconds": 120
}

GET /api/download_model/{job_id}
Response: Binary ONNX file (~10MB)
```

#### 3. WebContainer Settings API
```python
# Runs IN the browser WebContainer
POST /api/settings/save
{
    "anthropic_key": "sk-ant-...",
    "modal_token": "modal-token-...",
    "preferences": {
        "theme": "dark",
        "auto_load_last_scene": true
    }
}

GET /api/settings/load
Response:
{
    "keys_configured": true,
    "has_anthropic": true,
    "has_modal": true,
    "preferences": {...}
}
```

## 🛠️ Implementation Requirements

### 1. Scene Selector Component
```javascript
// Requirements:
- Grid view of available scenes with thumbnails
- Metadata display (robot type, DOF, description)
- "Upload Custom Scene" option
- Quick preview on hover
- Categories: Manipulation, Locomotion, Games
```

### 2. MuJoCo Integration
```javascript
// Requirements:
- Initialize WASM module
- Load XML + OBJ/MTL files
- Real-time rendering at 60 FPS
- Camera controls (orbit, zoom, pan)
- Reset simulation function
- Get observation data
- Apply action commands
```

### 3. WebContainer Setup
```python
# Requirements:
- Boot Python environment on page load
- Run Flask server on port 3000
- API proxy endpoints:
  - /llm/generate_policy
  - /modal/start_training  
  - /modal/training_status
- Inference endpoints:
  - /inference/load_model
  - /inference/predict
- Handle CORS for internal communication
```

### 4. LLM Integration
```python
# Modal function requirements:
@modal.function(gpu="A10G")
def generate_policy(
    prompt: str,
    scene_info: dict,  # Robot DOF, action space, etc.
    examples: list     # Few-shot examples
) -> dict:
    # Return:
    # - policy_code: str (Python RL code)
    # - explanation: str (What the policy does)
    # - hyperparams: dict (Suggested training params)
```

### 5. Training Pipeline
```python
# Modal function requirements:
@modal.function(gpu="A10G", timeout=3600)
def train_policy(
    policy_code: str,
    scene_xml: str,
    hyperparams: dict,
    websocket_url: str  # For progress updates
) -> str:
    # Return: URL to download ONNX model
    # Stream progress via websocket
```

### 6. UI Components

#### Prompt Interface
```
[                                          ] [Generate]
Suggestions: "Pick up objects" "Walk forward" "Balance"

[Advanced Options ▼]
  Learning Rate: [0.0003]
  Training Steps: [1M]
  Network Size: [64, 64]
```

#### Code Editor (Monaco)
- Syntax highlighting for Python
- Error indicators
- Collapsible sections
- "Validate" button
- "Reset to Generated" option

#### Training Monitor
```
Training "Pick up red block"...
━━━━━━━━━━━━━━━━━━━━ 45% • 2:30 remaining

Reward: ▁▂▃▄▅▆▇█ (improving!)
[Stop Early] [Download Current]
```

## 🎨 Design Guidelines

### Visual Style
- Clean, modern interface
- Dark mode by default
- Accent color: Electric blue (#2196F3)
- Monospace font for code
- Sans-serif for UI

### Layout
```
┌─────────────────────────────────────────┐
│ Header: Logo | Scene Selector | Settings │
├─────────────┬───────────────────────────┤
│             │                           │
│   Prompt    │      3D Viewport         │
│   Editor    │                           │
│   Code      │                           │
│   Params    │                           │
│             │                           │
├─────────────┴───────────────────────────┤
│ Status Bar: Training Progress | Messages │
└─────────────────────────────────────────┘
```

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