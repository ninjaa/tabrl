# Training Page Wireframe - Updated Design

## Overview
The training page allows users to select MuJoCo Playground models, describe desired behaviors in natural language, and train policies that are rendered as videos for comparison.

## Page Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TabRL Training Studio                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Model Selection                              │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │   │
│  │  │   Go1    │ │ ANYmal C │ │   UR5e   │ │  Franka  │ │ Allegro  │ │   │
│  │  │ [thumb]  │ │ [thumb]  │ │ [thumb]  │ │ [thumb]  │ │ [thumb]  │ │   │
│  │  │ Selected │ │          │ │          │ │          │ │          │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │   │
│  │                                                                      │   │
│  │  Model: Go1JoystickFlatTerrain (Quadruped Robot)                   │   │
│  │  Base Policy: ✓ Available (trained for 100k steps)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Behavior Description                            │   │
│  │                                                                      │   │
│  │  Describe the behavior you want:                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ "Keep shaking your hips aloha dude! Do a fun Hawaiian       │   │   │
│  │  │  dance with lots of hip movement"                            │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  Training Steps: [====|====] 50,000 (configurable)                 │   │
│  │                                                                      │   │
│  │  [Generate Training Approaches]                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Training Approaches                              │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐         │   │
│  │  │  Approach 1   │  │  Approach 2   │  │  Approach 3   │         │   │
│  │  │               │  │               │  │               │         │   │
│  │  │ Hip Oscillator│  │ Dance Rhythm  │  │ Full Body Sway│         │   │
│  │  │               │  │               │  │               │         │   │
│  │  │ Focus on hip  │  │ Rhythmic full │  │ Coordinate all│         │   │
│  │  │ joint swaying │  │ body movement │  │ joints in wave│         │   │
│  │  │               │  │               │  │               │         │   │
│  │  │ [Train This]  │  │ [Train This]  │  │ [Train This]  │         │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Training Progress                              │   │
│  │                                                                      │   │
│  │  Approach 1: [████████████████████] 100% Complete ✓               │   │
│  │  Approach 2: [████████████        ] 70%  Training...               │   │
│  │  Approach 3: [██                  ] 10%  Training...               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Side-by-Side Comparison                          │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐         │   │
│  │  │               │  │               │  │               │         │   │
│  │  │   [Video 1]   │  │   [Video 2]   │  │   [Video 3]   │         │   │
│  │  │               │  │               │  │               │         │   │
│  │  │  Hip Swaying  │  │ Dance Rhythm  │  │  Full Wave    │         │   │
│  │  │               │  │               │  │               │         │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘         │   │
│  │                                                                      │   │
│  │  [▶ Play All]  [⏸ Pause]  [↻ Loop]  [Download Videos]            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Model Selection
- **Visual Grid**: Show all available MuJoCo Playground models
- **Thumbnails**: Static render frames or mini preview videos
- **Selection State**: Clear indication of selected model
- **Base Policy Status**: Show if pre-trained base is available

### 2. Behavior Input
- **Natural Language**: Fun, creative prompts encouraged
- **Training Steps**: Slider to configure (10k - 100k range)
- **Default**: 50k steps for good balance of quality/speed

### 3. Training Approaches
- **LLM Generated**: 3 different reward function approaches
- **Clear Descriptions**: What each approach focuses on
- **Independent Training**: Each can be trained separately
- **Visual Cards**: Easy to understand layout

### 4. Training Progress
- **Real-time Updates**: WebSocket progress from Modal
- **Multiple Jobs**: Track all 3 training jobs
- **Time Estimates**: Show remaining time

### 5. Video Comparison
- **Side-by-Side**: All 3 videos play synchronously
- **10-second Clips**: Perfect length for comparison
- **Playback Controls**: Play/pause/loop functionality
- **Download Option**: Save favorite behaviors

## Technical Implementation

### Backend Requirements

1. **Model Listing Endpoint**
   ```python
   @app.get("/api/models/playground")
   def list_playground_models():
       return [
           {
               "id": "Go1JoystickFlatTerrain",
               "name": "Go1 Robot",
               "type": "quadruped",
               "thumbnail": "/static/thumbnails/go1.png",
               "base_policy": "go1_base_100k.pkl",
               "base_available": True
           },
           # ... more models
       ]
   ```

2. **Training Configuration**
   ```python
   @app.post("/api/training/configure")
   def configure_training(
       model_id: str,
       prompt: str,
       num_steps: int = 50000
   ):
       # Generate 3 reward approaches
       # Return training configurations
   ```

3. **Batch Training**
   ```python
   @app.post("/api/training/batch")
   def start_batch_training(
       model_id: str,
       approaches: List[RewardApproach],
       num_steps: int
   ):
       # Launch all 3 Modal training jobs
       # Return job IDs for tracking
   ```

4. **Video Generation**
   ```python
   @app.post("/api/training/render")
   def render_policy_video(
       model_id: str,
       policy_path: str,
       duration: float = 10.0
   ):
       # Use render_brax_model.py
       # Return video URL
   ```

### Frontend Requirements

1. **Model Gallery Component**
   - Grid layout with hover effects
   - Thumbnail loading (static images or GIFs)
   - Selection management

2. **Training Configurator**
   - Text input with examples
   - Slider for step count
   - Generate button

3. **Progress Tracker**
   - WebSocket connection for updates
   - Progress bars with percentages
   - Time remaining estimates

4. **Video Player Grid**
   - Synchronized playback
   - Individual video controls
   - Full-screen option

## User Flow

1. **Select Model**: User clicks on Go1 (or any model)
2. **Enter Prompt**: "Keep shaking your hips aloha dude!"
3. **Configure Steps**: Adjust slider to desired training length
4. **Generate Approaches**: LLM creates 3 reward functions
5. **Start Training**: Click "Train All" or individual approaches
6. **Monitor Progress**: Watch real-time training updates
7. **View Results**: Compare videos side-by-side
8. **Download Favorites**: Save best performing policies

## Benefits of This Approach

1. **No MuJoCo WASM Needed**: Videos are pre-rendered server-side
2. **Fast Iteration**: 30-60 second training for quick results
3. **Easy Comparison**: Side-by-side videos show differences
4. **Scalable**: Can train many approaches in parallel
5. **Shareable**: Videos can be downloaded and shared

## Next Steps

1. Update backend API endpoints
2. Create thumbnail generation script
3. Implement batch training in Modal
4. Build React components for the UI
5. Add video synchronization logic
