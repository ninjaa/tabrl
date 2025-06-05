# TabRL Action Plan - Demo Preparation

## 🚨 UPDATED STATUS (2:45 AM) - Demo at 5 PM Today!

### ✅ COMPLETED 
1. **End-to-End Training Pipeline**
   - Real MuJoCo RL training with PPO implemented
   - LLM reward generation integrated with training
   - Training progress tracking and model export
   - Frontend successfully trains policies (tested with "dance" task)

2. **Architecture Decision: Server-Side JAX Inference**
   - Removed all ONNX references from PRD and codebase
   - Server-side JAX inference via WebSocket confirmed
   - Avoids TensorFlow/tf2onnx dependency conflicts
   - MuJoCo WASM for visualization only

3. **Policy Generation Strategy**
   - Simplified to fine-tuning only (Level 2) for demo
   - 30-60 second training time target
   - Pre-trained base policies required
   - LLM generates Brax-compatible reward functions

### 🔄 CRITICAL PATH TO DEMO (Next 2-3 Hours)

#### 1. **Brax Integration for Modal Training** (HIGHEST PRIORITY)
- [ ] Update `modal_playground_training.py` to inject LLM reward functions
- [ ] Implement `compile_reward()` function that executes LLM code
- [ ] Add semantic API adapters (get_joint_angle → Brax state)
- [ ] Test with simple task like "walk forward"

#### 2. **Pre-trained Base Policies** (For Fine-tuning)
- [ ] Train base locomotion policies for each robot type
- [ ] Save JAX parameters to Modal volume
- [ ] Create policy loading mechanism for fine-tuning
- [ ] Test fine-tuning from base policy works

#### 3. **WebSocket Inference Implementation**
- [ ] Create `/api/models/{id}/inference` WebSocket endpoint
- [ ] Load JAX model from saved params
- [ ] Stream observations → actions in real-time
- [ ] Test latency and performance

#### 4. **Frontend WebSocket Integration**
- [ ] Add WebSocket client to React app
- [ ] Connect MuJoCo WASM sim to server inference
- [ ] Handle connection/reconnection logic
- [ ] Display inference status

### 📋 DEMO FLOW (Must Work by 9 AM)

1. **User selects robot scene** (e.g., Go1, ANYmal, UR5e)
2. **User enters task** (e.g., "make the robot dance")
3. **System generates reward functions** (2-3 approaches)
4. **User selects reward and starts training** 
5. **Fine-tuning runs on Modal GPU** (30-60 seconds)
6. **Trained policy streams to browser** via WebSocket
7. **Robot performs behavior** in MuJoCo WASM

### 🎯 SUCCESS METRICS

**Minimum Viable Demo:**
- [ ] One robot type working end-to-end (Go1 preferred)
- [ ] Fine-tuning completes in under 60 seconds
- [ ] Visible behavior change from natural language
- [ ] Stable WebSocket inference (no crashes)

**Nice to Have:**
- [ ] Multiple robot types (Go1, ANYmal, UR5e)
- [ ] Training progress visualization
- [ ] Multiple pre-trained behaviors to show
- [ ] Smooth 30+ FPS inference

### 🚀 IMMEDIATE NEXT STEPS

```bash
# 1. Test current Modal deployment
cd backend && modal deploy modal_playground_training.py

# 2. Implement reward injection in Modal function
# Edit modal_playground_training.py to accept reward_code parameter

# 3. Create base policy training script
# Train and save base policies for fine-tuning

# 4. Implement WebSocket inference endpoint
# Add to app.py with JAX model loading
```

### 📝 Key Technical Decisions

1. **No ONNX Export** - Server-side JAX inference only
2. **Fine-tuning Only** - Skip command-based and full training
3. **Pre-trained Base** - Required for 30-60 second training
4. **Brax Environments** - Use registry or custom wrapper
5. **WebSocket Streaming** - For real-time inference

### ⏰ TIME ALLOCATION (2:45 AM - 9:00 AM)

- **2:45-4:00 AM**: Brax reward injection + Modal training
- **4:00-5:00 AM**: WebSocket inference implementation  
- **5:00-6:00 AM**: Frontend WebSocket integration
- **6:00-7:00 AM**: Pre-trained base policies
- **7:00-8:00 AM**: Integration testing
- **8:00-9:00 AM**: Demo polish and practice

### 🔥 KNOWN BLOCKERS

1. **Reward Function Format** - Need to adapt LLM code to Brax
2. **Base Policy Training** - Takes time, start ASAP
3. **WebSocket Latency** - Must be < 50ms for smooth control
4. **Modal GPU Access** - Ensure credits/quotas available

---

## Post-Demo Improvements

1. **Progressive Training System** - Restore 3-tier approach
2. **ONNX Export Pipeline** - Separate container for conversion  
3. **Browser Inference** - True client-side execution
4. **More Environments** - Full MuJoCo Playground integration
5. **Sim-to-Real** - Hardware deployment examples
{{ ... }}
