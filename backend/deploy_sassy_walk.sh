#!/bin/bash
# Deploy sassy walk training to Modal

echo "🚀 Deploying sassy walk training to Modal..."

# First, let's test with a short training run (500k steps instead of 5M for demo)
modal run modal_custom_reward_training.py \
    --policy-json hip_swing_policy.json \
    --reward-index 0 \
    --num-timesteps 500000 \
    --output sassy_walk_demo.pkl

echo "✅ Training deployed!"
