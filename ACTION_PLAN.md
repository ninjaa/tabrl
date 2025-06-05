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
