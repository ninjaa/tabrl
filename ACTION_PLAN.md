# TabRL Action Plan - Updated Progress

## COMPLETED: Backend & Frontend Integration
1. [x] Create project structure
2. [x] Run `./scripts/setup_scenes.sh` to get robot XMLs  
3. [x] Set up Modal account and get token
4. [x] **FastAPI Backend Implementation**
   - [x] Multi-LLM support (Claude, GPT-4, Gemini, DeepSeek)
   - [x] Scene parsing and robot structure analysis
   - [x] Policy generation with JSON schema validation
   - [ ] Real RL training pipeline with PPO
   - [ ] ONNX model export and management
5. [x] **React Frontend Implementation**
   - [x] Dynamic scene and model loading from API
   - [x] Policy generation UI with reward function display
   - [x] Training progress visualization
   - [x] Modern styling and responsive design
6. [x] **End-to-End Pipeline Validation**
   - [ ] Successful training completion (dance task, 200+ reward)
   - [ ] Model export and persistence
   - [ ] Progress tracking and status updates

## IN PROGRESS: Critical Fixes
1. [x] ~~Reward function JSON parsing (triple-quote issue)~~ 
2. [ ] **Reward function formatting** - Functions not properly wrapped
   - [x] Updated LLM prompt to specify function definitions
   - [ ] Test with new prompt and verify function wrapping
   - [ ] Add validation for proper function syntax
3. [ ] **UI Code Display Issues**
   - [x] Fixed text alignment (left-aligned code blocks)
   - [ ] Test code formatting with proper functions
   - [ ] Add syntax highlighting for Python code

## NEXT PHASE: Brax/JAX Integration for Robust Locomotion

### Phase 3A: Local Development (CPU) - JAX/Brax Setup
1. [ ] **Add JAX/Brax Dependencies**
   - [ ] Update backend requirements: `jax`, `brax`, `mujoco-mjx`, `flax`
   - [ ] Test JAX CPU installation and basic functionality
   - [ ] Create brax environment loader for existing scenes

2. [ ] **Brax Environment Wrapper**
   - [ ] Create `BraxEnvironmentWrapper` that loads our MuJoCo XMLs
   - [ ] Adapt semantic reward APIs to work with Brax state format
   - [ ] Test dimension consistency (obs_dim, action_dim) with original scenes
   - [ ] Integrate LLM-generated reward functions into Brax environments

3. [ ] **ONNX Export Pipeline**
   - [ ] Implement JAX/Flax → ONNX conversion utility
   - [ ] Test with simple policy networks first
   - [ ] Validate input/output dimensions match MuJoCo WASM expectations
   - [ ] Create export endpoint: `/api/models/{id}/export/onnx`

### Phase 3B: Modal GPU Training Integration
1. [ ] **Modal Training Service**
   - [ ] Create Modal function for Brax PPO training
   - [ ] Configure GPU requirements and dependencies
   - [ ] Set up training job submission and status tracking
   - [ ] Test with ANYmal locomotion task (7-minute target)

2. [ ] **Training Pipeline Integration**
   - [ ] Update `/api/training/start` to use Modal Brax training
   - [ ] Implement progress streaming from Modal to frontend
   - [ ] Add automatic ONNX export after training completion
   - [ ] Test full pipeline: LLM rewards → Brax training → ONNX export

### Phase 3C: Frontend ONNX Inference
1. [ ] **MuJoCo WASM + ONNX.js Integration**
   - [ ] Set up ONNX.js in React frontend
   - [ ] Load trained policies and run inference in browser
   - [ ] Test with exported locomotion policies
   - [ ] Implement real-time simulation visualization

2. [ ] **UI Enhancement**
   - [ ] Add policy comparison viewer
   - [ ] Training progress visualization (like MuJoCo Playground)
   - [ ] Export/share trained policies functionality

## IMMEDIATE PRIORITIES (This Session)

### 1. JAX/Brax Local Setup
Test JAX installation and basic Brax functionality on CPU:
```bash
cd backend
uv add jax brax mujoco-mjx flax
uv run python -c "import jax; import brax; print('JAX/Brax ready!')"
```

### 2. Create Brax Environment Wrapper
Build bridge between our MuJoCo scenes and Brax training:
- [ ] Load ANYmal scene in Brax format
- [ ] Test observation/action dimensions match current setup
- [ ] Integrate one LLM reward function for testing

### 3. ONNX Export Prototype
Create basic JAX → ONNX conversion:
- [ ] Simple policy network export
- [ ] Dimension validation
- [ ] Test in ONNX.js environment

## Commands Ready to Run:
```bash
# Test current fixes
cd tabrl/backend && uv run python app.py
cd tabrl/frontend && npm run dev

# Test policy generation with G1 humanoid
curl -X POST http://localhost:8000/api/policy/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Make the robot dance", "scene_name": "humanoids/unitree_g1", "model": "claude-sonnet-4-20250514"}'

```

# 🚀 TabRL Action Plan - HACKATHON MODE

## ⏰ **5-HOUR DEMO TIMELINE** 
*Current Time: 11:20 PM - Demo: 4:20 AM*

### **PHASE 1: Training Pipeline (NEXT 2 HOURS) - 🔥 IN PROGRESS**

#### ✅ **COMPLETED** (Last 4 hours):
- **Modal GPU deployment** - ✅ LIVE on H100 GPUs (note: H100 uses jax[cuda12_pip], A100 uses jax[cuda11_pip])
- **52 MuJoCo Playground environments discovered** - ✅ READY
- **Hero robot selection** - ✅ Go1, Berkeley Humanoid, Spot, Aloha, Panda
- **Intelligent policy generation API design** - ✅ DOCUMENTED
- **3-tier training approach** - ✅ Command-based → Fine-tuning → Full training

#### 🔥 **NOW TESTING** (11:20 PM):
- **Modal training execution** - 🧪 Go1JoystickFlatTerrain 100K steps
- **JAX/Flax to ONNX conversion pipeline** - ⏳ NEXT
- **Model persistence on Modal volumes** - ⏳ TESTING

#### 🎯 **NEXT 2 HOURS** (11:20 PM - 1:20 AM):
1. **Verify Modal training works** (30 min)
2. **Add ONNX export to training pipeline** (45 min)  
3. **Test full training → ONNX → download flow** (45 min)

### **PHASE 2: Scene Bridge (1:20 AM - 2:20 AM)**

#### 🎯 **The Mind-Blowing Connection**:
```python
# Backend: Extract XML from any Playground environment
env = registry.locomotion.load("Go1JoystickFlatTerrain") 
xml_string = env.sys.to_xml()  # Complete scene XML!

# Frontend: Load directly in MuJoCo WASM
mujoco.Model.load_from_xml(xml_string)
# 🤯 User sees Go1 robot in full outdoor terrain!
```

#### ✅ **Tasks**:
1. **XML Export API** - `/api/playground/{category}/{env}/xml` (15 min)
2. **Frontend scene selector dropdown** (30 min)
3. **MuJoCo WASM integration** (30 min)  
4. **Test: Pick environment → See robot in scene** (15 min)

### **PHASE 3: End-to-End Demo (2:20 AM - 4:20 AM)**

#### 🎬 **Demo Flow**:
1. **User selects**: "Go1 Quadruped in Park"
2. **Scene loads**: Complete outdoor terrain with Go1 robot
3. **User prompts**: "Make robot dance to electronic music"
4. **AI generates**: Custom reward function for dancing behavior  
5. **Modal trains**: 30-60 second fine-tuning on H100 GPU
6. **Policy downloads**: ONNX model ready for browser
7. **Robot dances**: Live in browser with trained behavior!

#### 🎯 **Demo Scenarios**:
- **Go1**: "Do parkour jumps" / "Dance to music" / "Follow patrol route"
- **Berkeley Humanoid**: "Wave hello" / "Do martial arts" / "Play catch"
- **Spot**: "Inspect construction site" / "Security patrol" / "Industrial walk"

### **PHASE 4: Polish & Presentation (4:20 AM)**

#### 🎨 **Demo Ready Features**:
- ✅ **52 environments** available in dropdown
- ✅ **3-tier intelligent training** (0s → 60s → 8min)
- ✅ **Live robot visualization** in browser  
- ✅ **Real-time policy generation** from natural language
- ✅ **Cloud GPU training** on Modal infrastructure

---

## 🔧 **Technical Status**

### **Backend Infrastructure**:
- ✅ **Modal GPU Service**: Deployed on H100 hardware (CUDA 12)
- ✅ **MuJoCo Playground**: 52 environments integrated
- ✅ **JAX/Brax Training**: PPO with domain randomization
- 🧪 **ONNX Export**: JAX → TensorFlow → ONNX pipeline
- ✅ **Volume Storage**: Persistent model storage on Modal

### **Frontend Integration**:
- ✅ **MuJoCo WASM**: Existing physics simulation  
- ⏳ **Scene Loading**: XML export from Playground environments
- ⏳ **ONNX Inference**: Browser-based policy execution
- ⏳ **Real-time Control**: Policy outputs → robot movements

### **AI Pipeline**:
- ✅ **LLM Integration**: Claude, GPT-4, Gemini via LiteLLM
- ✅ **Reward Generation**: Natural language → custom reward functions
- ✅ **Scene Analysis**: MuJoCo XML parsing for context
- ⏳ **Policy Optimization**: 3-tier training approach

---

## 🎯 **Success Metrics**

### **Must Have (Demo Requirements)**:
- [ ] User picks robot environment from dropdown
- [ ] User sees robot in complete 3D scene  
- [ ] User types behavior request: "make robot dance"
- [ ] System trains policy in under 60 seconds
- [ ] Robot performs trained behavior in browser
- [ ] End-to-end: 5 minutes from prompt to dancing robot

### **Nice to Have (Demo Wow Factors)**:
- [ ] Multiple robot types working (Go1, Humanoid, Spot)
- [ ] Complex behaviors (parkour, martial arts, inspection)
- [ ] Training progress visualization 
- [ ] Policy comparison between approaches
- [ ] Real-time training metrics display

---

## 🚨 **Critical Path Dependencies**

### **Blocker Resolution**:
1. **Modal training success** - 🧪 TESTING NOW
2. **ONNX conversion working** - ⏳ IMPLEMENTING  
3. **XML export functional** - ⏳ 15 MIN TASK
4. **Frontend scene loading** - ⏳ 30 MIN TASK

### **Risk Mitigation**:
- **Backup Plan A**: Use command-based generation (0 seconds) if training fails
- **Backup Plan B**: Demo with existing MuJoCo scenes if XML export fails  
- **Backup Plan C**: Show training progress only if ONNX conversion fails

**Current Status: 🟡 ON TRACK** for 5-hour demo completion!

---

## 🚀 Quick Reference Commands

### Modal GPU Deployment
```bash
# Deploy/update Modal app
cd backend && MODAL_FORCE_BUILD=1 uv run modal deploy modal_playground_training.py

# Test training (100K steps, ~30s on H100)
uv run python playground_trainer.py --quick-test --env Go1JoystickFlatTerrain --category locomotion

# Full training (30M steps, ~1 min on H100)
uv run python playground_trainer.py --env Go1JoystickFlatTerrain --category locomotion --steps 30000000
```

### Local Development
```bash
# Backend
cd backend && uv run python app.py

# Frontend
cd frontend && npm run dev
```

---

## 🔮 Future Improvements (Post-Hackathon)

### Infrastructure Optimizations
- [ ] **Split ONNX export to separate container with its own dependencies**
  - Avoid JAX/TensorFlow/tf2onnx dependency conflicts
  - Use CPU-only container for conversion (cost savings)
  - Cleaner separation of concerns between training and export

### Additional Features
- [ ] Multi-agent training scenarios
- [ ] Sim-to-real transfer validation
- [ ] Custom reward function library
- [ ] Training hyperparameter optimization
- [ ] Model versioning and A/B testing

## TabRL Action Plan for AI Engineer World's Fair Demo

## 🚨 Demo Time: 5 PM Today! Core functionality needed by 9 AM!

### Current Status (1:30 AM)
- ✅ JAX downgraded to 0.6.0 to fix cuSolver GPU error  
- ✅ Removed TensorFlow/tf2onnx due to dependency conflicts
- ✅ Created server-side JAX inference system as alternative
- ✅ Added playground API endpoints for scene listing/retrieval
- 🔄 Need to test Modal deployment with simplified dependencies
- 🔄 Need to implement frontend WebSocket client for inference

### Immediate Actions (Demo-Ready by 9 AM)

1. **Deploy Updated Modal Container** (1:30-2:00 AM)
   - [x] Remove numpy pin to resolve dependencies
   - [x] Remove TensorFlow/tf2onnx/ONNX from container
   - [x] Test deployment: `cd backend && modal deploy modal_playground_training.py`
   - [x] Verify GPU training works with JAX 0.6.0

2. **Test Server-Side Inference** (2:00-2:30 AM)
   - [ ] Create test endpoint to load a trained model
   - [ ] Test WebSocket inference with dummy observations
   - [ ] Measure latency for real-time control

3. **Update Frontend for Server-Side Inference** (2:30-3:30 AM)
   - [ ] Create WebSocket client for policy inference
   - [ ] Integrate with MuJoCo WASM simulation loop
   - [ ] Add connection status indicator
   - [ ] Handle reconnection logic

4. **Integration Testing** (3:30-4:00 AM)
   - [ ] Train a simple locomotion policy
   - [ ] Load it via inference API
   - [ ] Test full loop: MuJoCo sim → server inference → actions → sim

5. **Demo Polish** (4:00-5:00 AM)
   - [ ] Create demo scene selection UI
   - [ ] Add training progress visualization
   - [ ] Prepare 2-3 trained policies for instant demo
   - [ ] Test end-to-end flow multiple times

### Post-Demo Improvements

1. **Split Container Architecture**
   - Create separate Modal container for ONNX export
   - Use older numpy/TF versions compatible with tf2onnx
   - Export trained JAX models to ONNX in batch process

2. **Direct JAX → ONNX Export**
   - Investigate jax2onnx or similar tools
   - Bypass TensorFlow entirely if possible

3. **Client-Side JAX**
   - Explore running JAX in browser via WASM
   - Would eliminate server-side inference need

### Dependencies Summary

**Training Container (Modal GPU)**:
- Python 3.10 base
- jax[cuda12]==0.6.0 (for H100 GPU)
- mujoco, mujoco_mjx, brax, playground
- flax, optax, numpy (unpinned)
- NO TensorFlow/tf2onnx/ONNX

**Backend API**:
- FastAPI with WebSocket support
- JAX for model loading/inference
- Standard backend dependencies

**Frontend**:
- React + MuJoCo WASM
- WebSocket client for inference
- onnxruntime-web (for future use)

### Key Architecture Change
Due to tf2onnx/TensorFlow/numpy dependency conflicts, we're using:
- **Server-side inference** via WebSocket for the demo
- JAX models stay on backend, frontend sends observations
- Post-demo: implement proper ONNX export pipeline

This approach ensures demo stability while preserving path to browser-based inference.
