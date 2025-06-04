# TabRL Implementation Plan

## Day 1: Infrastructure Setup ✓
1. [x] Create project structure
2. [ ] Run `./scripts/setup_scenes.sh` to get robot XMLs
3. [ ] Set up Modal account and get token
4. [ ] Deploy hello world Modal function
5. [ ] Test WebContainer boots and saves settings
6. [ ] Verify Modal proxy works (no CORS!)

## Day 2: Core Integration
1. [ ] Get MuJoCo WASM from co-hacker
2. [ ] Load a simple scene (cartpole)
3. [ ] Extract observations generically
4. [ ] Test ONNX inference in WebContainer

## Day 3: Training Pipeline
1. [ ] Implement generic training env
2. [ ] Test reward function generation
3. [ ] Run minimal training job
4. [ ] Export and download ONNX

## Day 4: Polish & Demo
1. [ ] Scene selector UI
2. [ ] Pretty 3D viewer
3. [ ] Training progress viz
4. [ ] Ship it!

## Commands to Run Now:
```bash
cd tabrl
./scripts/setup_scenes.sh
cd frontend && npm install
npm run dev

# In another terminal:
cd tabrl/modal
modal setup
modal serve hello_world.py
```
