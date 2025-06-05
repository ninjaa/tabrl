# Training Page V2 Design Document

## Overview
The new training page redesign focuses on parallel multi-LLM training with video comparison, leveraging Modal's deployed GPU functions for speed.

## Key Features

### 1. Model Gallery Section
- **Visual Grid Layout**: Thumbnail previews of all available robots
- **Metadata Display**: Category, DOF, pre-trained badge
- **Smart Defaults**: Auto-selects Go1 (has base policy) 
- **Responsive Design**: Adapts to screen size

### 2. Behavior Configuration
- **Natural Language Input**: Fun prompts like "Keep shaking your hips aloha dude! 🌺"
- **Training Steps Slider**: 10K-100K steps with time estimates
  - Quick: 10K steps (~30 seconds)
  - Standard: 50K steps (~2 minutes)
  - Quality: 100K steps (~4 minutes)
- **Single Button**: Generate approaches for all 3 LLMs simultaneously

### 3. Multi-LLM Training Tabs
- **Three Provider Tabs**:
  - Claude Opus 3.5
  - OpenAI GPT-4
  - Gemini Pro 2.5
- **Per-Tab Content**:
  - 3 reward approach cards with descriptions
  - Expandable code preview
  - "Train All 3 Approaches" button
  - Real-time progress bars for each job

### 4. Video Comparison Grid
- **9-Video Grid**: 3 LLMs × 3 approaches each
- **Synchronized Playback**: Play/pause/reset all videos
- **Per-Video Info**: Approach name, training stats
- **Auto-Loop**: Videos loop for easy comparison

## Technical Implementation

### Frontend Components
```javascript
TrainingPageV2.jsx
├── Model Gallery (GET /api/playground/models)
├── Behavior Config (training parameters)
├── LLM Tabs (approach generation)
├── Training Progress (job polling)
└── Video Grid (synchronized playback)
```

### Backend Endpoints
```python
# New endpoints in app.py:
GET  /api/playground/models      # List models with metadata
POST /api/training/approaches    # Generate approaches per LLM
POST /api/training/batch        # Start parallel Modal jobs
GET  /api/training/{id}/status  # Poll job progress
POST /api/training/render       # Generate videos
```

### Modal Integration
- **Deployed Function**: `tabrl-custom-reward-training`
- **Parallel Execution**: `train_fn.spawn()` for async jobs
- **Progress Tracking**: Modal job status polling
- **Video Generation**: Automatic post-training

## User Flow

1. **Select Robot**: Click model card from visual gallery
2. **Describe Behavior**: Enter natural language prompt
3. **Set Training Steps**: Adjust slider for speed/quality tradeoff
4. **Generate Approaches**: Single click generates for all 3 LLMs
5. **Review & Train**: See approaches per LLM, train all with one click
6. **Monitor Progress**: Real-time progress bars per job
7. **Compare Results**: 9-video grid with synchronized playback

## Performance Optimizations

### Speed Improvements
- **Modal Deployment**: Pre-deployed function eliminates cold starts
- **Parallel Training**: 9 jobs run simultaneously on H100 GPUs
- **Batch API Calls**: Single request starts 3 jobs per LLM
- **Efficient Polling**: 2-second intervals with progress estimation

### UX Enhancements
- **Visual Feedback**: Loading states, progress bars, status badges
- **Smart Defaults**: Pre-selected models, reasonable step counts
- **Error Recovery**: Graceful handling of failed jobs
- **Responsive Design**: Works on desktop and tablet

## Future Enhancements

1. **Advanced Features**:
   - Domain randomization toggle
   - Hyperparameter tuning UI
   - Custom reward code editor
   - Training curves visualization

2. **Social Features**:
   - Share trained policies
   - Leaderboard for tasks
   - Community reward functions
   - Vote on best approaches

3. **Export Options**:
   - Download trained models
   - Export videos as GIF
   - Generate training reports
   - ONNX model conversion

## Design Rationale

### Why Multi-LLM?
- **Diversity**: Different LLMs generate unique approaches
- **Comparison**: See which provider works best for tasks
- **Reliability**: If one fails, others still work
- **Research**: Valuable data on LLM robotics capabilities

### Why Video-Based?
- **Accessibility**: No WASM/physics knowledge needed
- **Shareability**: Easy to show results to others
- **Performance**: Server-side rendering is consistent
- **Comparison**: Side-by-side video is intuitive

### Why This Layout?
- **Progressive Disclosure**: Steps reveal as needed
- **Visual First**: Gallery and videos dominate
- **Minimal Clicks**: Batch operations reduce interactions
- **Clear Hierarchy**: Important actions are prominent

This design positions TabRL as the premier platform for AI-powered robot training, making it accessible to beginners while powerful enough for researchers.
