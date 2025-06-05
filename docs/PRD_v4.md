# MuJoCo Browser Training Platform - PRD

## 🎯 Product Vision
Build a browser-based robotics training platform where users can teach robots new behaviors using natural language. The system generates multiple training approaches, trains them in parallel on GPUs, and presents side-by-side video comparisons of the results.

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Frontend (React)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Model     │  │   Training   │  │    Video        │    │
│  │  Gallery    │  │   Progress   │  │   Comparison    │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ API Calls + WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │     LLM     │  │   Training   │  │   Video         │    │
│  │ Integration │  │   Batch API  │  │   Generation    │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ GPU Training
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Modal (Serverless GPU)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Brax/JAX    │  │   Model      │  │    Parallel     │    │
│  │  Training   │  │   Storage    │  │    Jobs         │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Stack

### Backend (FastAPI)
- **Purpose**: LLM integration, training orchestration, video generation
- **Key Features**:
   - Multi-provider LLM support (Claude, GPT-4, Gemini, DeepSeek)
   - Reward function generation from natural language
   - MuJoCo Playground model management
   - Batch training job orchestration
   - Server-side video rendering

### Frontend (React + Vite)
- **Purpose**: Model selection and video comparison interface
- **Key Features**:
   - Visual model gallery with thumbnails
   - Natural language task input
   - Training progress monitoring
   - Side-by-side video comparison
   - Download and sharing capabilities

### Training (Modal + Brax)
- **Purpose**: GPU-accelerated RL training
- **Key Features**:
   - Brax PPO training with JAX acceleration
   - Parallel training of multiple approaches
   - H100 GPUs for fast iteration (30s-2min)
   - Model storage in Modal volumes
   - Automatic video generation post-training

## 🎬 Core User Flow

1. **Select Model**: Choose from MuJoCo Playground gallery
2. **Describe Behavior**: Enter fun, creative prompts
3. **Configure Training**: Adjust steps for speed vs quality
4. **Generate Approaches**: LLM creates 3 different reward strategies
5. **Parallel Training**: All 3 approaches train simultaneously
6. **Compare Results**: Watch side-by-side videos of trained behaviors
7. **Download Favorites**: Save and share the best results

## 🏗️ Implementation Phases

### Phase 1: Model Gallery & Selection
- List all MuJoCo Playground environments
- Generate static thumbnails/previews
- Highlight available base policies
- Model metadata and descriptions

### Phase 2: Training Pipeline
- LLM reward generation with multiple approaches
- Batch job submission to Modal
- Progress tracking via WebSocket
- Configurable training steps

### Phase 3: Video System
- Server-side rendering with Brax
- 10-second behavior demonstrations
- Synchronized playback controls
- Download functionality

### Phase 4: Polish & Scale
- Pre-trained behavior library
- Training time estimates
- Queue management for GPU resources
- Social sharing features

## 🧠 Smart Training Approaches

Generate diverse robot behaviors using parallel training:

```python
# User prompt
"Keep shaking your hips aloha dude!"

# LLM generates 3 approaches
approaches = [
    {
        "name": "Hip Oscillator",
        "description": "Focus on rhythmic hip joint movements",
        "reward_code": "..."
    },
    {
        "name": "Dance Rhythm", 
        "description": "Full body coordination with beat",
        "reward_code": "..."
    },
    {
        "name": "Hawaiian Sway",
        "description": "Smooth, flowing whole-body motion",
        "reward_code": "..."
    }
]
```

## 🎯 Training Configuration

### Fine-tuning (with base policy)
- **Quick**: 10k steps (~30 seconds)
- **Standard**: 50k steps (~2 minutes) - Default
- **Quality**: 100k steps (~4 minutes)

### From Scratch
- **Minimum**: 100k steps (~4 minutes)
- **Standard**: 250k steps (~10 minutes)
- **High Quality**: 500k steps (~20 minutes)

## 📊 Key Features

### Parallel Training
```python
@app.post("/api/training/batch")
async def train_batch(request: BatchTrainingRequest):
    jobs = []
    for approach in request.approaches:
        job = modal.Function.lookup(
            "tabrl-training", 
            "train_playground_policy"
        ).spawn(
            env_name=request.model_id,
            reward_code=approach.code,
            num_timesteps=request.num_steps,
            use_pretrained=request.use_base_policy
        )
        jobs.append({
            "job_id": job.object_id,
            "approach": approach.name
        })
    return {"jobs": jobs}
```

### Video Generation
```python
@app.post("/api/training/render")
async def render_policy(request: RenderRequest):
    # Use render_brax_model.py
    video_path = render_brax_model(
        model_path=request.model_path,
        output_prefix=f"training_videos/{request.job_id}",
        episode_length=500  # 10 seconds at 50Hz
    )
    return {"video_url": f"/videos/{request.job_id}.mp4"}
```

## 🎯 Success Metrics

- **Training time**: < 2 minutes for standard fine-tuning
- **Parallel jobs**: Support 3+ simultaneous trainings
- **Video quality**: Smooth 50 FPS, 10-second clips
- **User engagement**: 80%+ complete full flow
- **LLM success**: 90%+ valid reward functions

## 📊 Example Behaviors

### Creative Prompts to Showcase
1. **"Do the robot dance from the 80s"** - Stiff, mechanical moves
2. **"Pretend you're a happy puppy"** - Excited jumping
3. **"Walk like you're on the moon"** - Low gravity gait
4. **"Breakdance with style"** - Dynamic spinning
5. **"March like a soldier"** - Precise, rhythmic steps

## 🔄 API Endpoints

### Model Management
```
GET  /api/playground/models      # List all available models
GET  /api/playground/{id}/info   # Model details and capabilities
GET  /api/playground/{id}/thumbnail  # Static preview image
```

### Training Pipeline  
```
POST /api/training/approaches    # Generate reward approaches
POST /api/training/batch         # Start parallel training jobs
GET  /api/training/{job_id}/status  # Check training progress
POST /api/training/render        # Generate result video
```

### Video Access
```
GET  /api/videos/{video_id}      # Stream video file
POST /api/videos/download        # Generate download link
```

## 🏃 Getting Started

```bash
# Backend setup
cd backend
pip install -r requirements.txt
modal deploy modal_playground_training.py
uvicorn app:app --reload

# Frontend setup  
cd frontend
npm install
npm run dev
```

## 📚 Key Technologies

- **MuJoCo Playground**: Pre-built robot environments
- **Brax**: Hardware-accelerated physics simulation
- **JAX**: High-performance ML computation
- **Modal**: Serverless GPU infrastructure
- **FastAPI**: Modern Python web framework
- **React**: Interactive frontend framework

## 🧪 Testing Checklist

### Backend Tests
- [ ] LLM generates valid reward functions
- [ ] Batch training launches successfully
- [ ] Video generation completes without errors
- [ ] Progress updates stream correctly

### Frontend Tests  
- [ ] Model gallery loads all environments
- [ ] Training configuration UI works smoothly
- [ ] Videos play synchronously side-by-side
- [ ] Download functionality works

### Integration Tests
- [ ] End-to-end: prompt → training → video comparison
- [ ] Multiple concurrent users supported
- [ ] System handles training failures gracefully
- [ ] Videos render consistently across models