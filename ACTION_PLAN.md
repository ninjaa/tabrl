# TabRL Action Plan - Training Page Focus

## 🎯 New Direction: Video-Based Training Comparison

### Core Concept
- **Cut MuJoCo WASM** for now - focus on server-rendered videos
- **Show all Playground models** with Go1 as the hero (already have base policy)
- **Train 3 approaches** from each prompt and show videos side-by-side
- **Make it fun** - encourage creative, humorous prompts

### ✅ What We Have Working
1. **Brax PPO Training** via Modal GPU (H100)
2. **Base Policy for Go1** (100k steps trained)
3. **Video Rendering** (`render_brax_model.py` creates 10-second videos)
4. **LLM Reward Generation** (fixed JSON parsing bug)

### 🎯 Immediate Tasks for Training Page

#### 1. Backend API Updates
- [ ] Create `/api/playground/models` endpoint listing all available models
- [ ] Generate static thumbnails for each model
- [ ] Update `/api/training/approaches` to accept `num_steps` parameter
- [ ] Create `/api/training/batch` endpoint to launch 3 Modal jobs
- [ ] Add `/api/training/render-video` endpoint using our render script

#### 2. Model Thumbnails
- [ ] Create script to generate static preview images
- [ ] Consider: Single frame at interesting pose vs mini GIF
- [ ] Store in `backend/static/model_previews/`
- [ ] Include model metadata (type, DOF, description)

#### 3. Training Configuration
- **Default Steps**: 50,000 (1-2 minutes on H100)
- **Range**: 10k (quick test) to 100k (high quality)
- **Fine-tuning**: Start from base policy for Go1
- **From Scratch**: Other models without base policies

#### 4. Batch Training System
```python
# Launch 3 training jobs in parallel
def train_multiple_approaches(
    model_id: str,
    approaches: List[RewardApproach], 
    num_steps: int,
    use_base_policy: bool
):
    job_ids = []
    for approach in approaches:
        job_id = modal.Function.lookup(
            "tabrl-training", 
            "train_playground_policy"
        ).spawn(
            env_name=model_id,
            reward_code=approach.code,
            num_timesteps=num_steps,
            use_pretrained=use_base_policy
        )
        job_ids.append(job_id)
    return job_ids
```

#### 5. Video Generation Pipeline
- [ ] After training completes, auto-generate 10s video
- [ ] Store videos in `backend/training_videos/`
- [ ] Return video URLs for frontend playback
- [ ] Ensure videos are synchronized (same start frame)

### 📐 Frontend Components Needed

#### 1. Model Selector Grid
```jsx
<ModelGrid>
  {models.map(model => (
    <ModelCard 
      key={model.id}
      selected={selectedModel === model.id}
      thumbnail={model.thumbnail}
      name={model.name}
      hasBasePolicy={model.base_available}
      onClick={() => setSelectedModel(model.id)}
    />
  ))}
</ModelGrid>
```

#### 2. Training Configurator
```jsx
<TrainingConfig>
  <PromptInput 
    placeholder="Keep shaking your hips aloha dude!"
    value={prompt}
    onChange={setPrompt}
  />
  <StepSlider
    min={10000}
    max={100000}
    step={10000}
    value={numSteps}
    onChange={setNumSteps}
    labels={{
      10000: "Quick (30s)",
      50000: "Standard (2min)",
      100000: "Quality (4min)"
    }}
  />
</TrainingConfig>
```

#### 3. Video Comparison View
```jsx
<VideoGrid>
  {videos.map((video, idx) => (
    <VideoPlayer
      key={idx}
      src={video.url}
      title={approaches[idx].name}
      description={approaches[idx].description}
      synchronized={true}
      loop={true}
    />
  ))}
</VideoGrid>
```

### 🚀 Execution Plan

#### Phase 1: API & Backend (2 hours)
1. List playground models endpoint
2. Batch training endpoint  
3. Video generation integration
4. Progress tracking via WebSocket

#### Phase 2: Model Assets (1 hour)
1. Generate thumbnails for all models
2. Create model metadata JSON
3. Identify which models have base policies

#### Phase 3: Frontend UI (2 hours)
1. Model selector component
2. Training configurator with slider
3. Approach cards with descriptions
4. Progress tracking display

#### Phase 4: Video System (1 hour)
1. Video player component
2. Synchronization logic
3. Download functionality
4. Side-by-side layout

### 📊 Training Time Recommendations

For **Fine-tuning** (with base policy):
- **Quick Test**: 10k steps (~30 seconds)
- **Good Result**: 25k steps (~1 minute)  
- **Best Quality**: 50k steps (~2 minutes)

For **From Scratch** (no base policy):
- **Minimum**: 100k steps (~4 minutes)
- **Recommended**: 250k steps (~10 minutes)
- **High Quality**: 500k steps (~20 minutes)

### 🎨 Example Prompts to Showcase

1. **"Keep shaking your hips aloha dude!"** - Hip-focused hula dance
2. **"Do the robot dance from the 80s"** - Stiff, mechanical movements
3. **"Pretend you're a happy puppy"** - Excited jumping and tail wagging
4. **"Walk like you're on the moon"** - Low gravity locomotion
5. **"Breakdance with style"** - Spinning and dynamic moves

### ⚡ Quick Wins

1. **Static Demos**: Pre-train some fun behaviors to show immediately
2. **Canned Responses**: Have backup reward functions if LLM fails
3. **Progress Animations**: Make training feel faster with good UI
4. **Sound Effects**: Add audio feedback for completed training

### 🔧 Technical Decisions

1. **No MuJoCo WASM**: Simplifies frontend, videos work everywhere
2. **Parallel Training**: User sees all 3 approaches training at once
3. **Video Format**: MP4 with H.264 for universal playback
4. **Configurable Steps**: Let users trade time for quality

### ✅ Success Criteria

- [ ] User can select any playground model
- [ ] Generate 3 approaches from a fun prompt
- [ ] All 3 train in parallel (under 2 minutes for Go1)
- [ ] Videos play side-by-side for comparison  
- [ ] Can download favorite videos
- [ ] Works reliably without crashes

### 🚨 Potential Issues & Solutions

**Issue**: Training takes too long
**Solution**: Default to fewer steps, show progress clearly

**Issue**: Videos don't sync properly  
**Solution**: Ensure all videos start from same initial state

**Issue**: LLM generates invalid rewards
**Solution**: Validate and have fallback rewards ready

**Issue**: Modal GPU unavailable
**Solution**: Queue system with estimated wait times
