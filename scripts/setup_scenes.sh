#!/bin/bash
# Download a few good starter scenes

mkdir -p scenes/{simple,manipulation,locomotion}

# 1. Simple control tasks (fast training)
git clone https://github.com/google-deepmind/mujoco_menagerie temp_menagerie

# Copy just a few good ones
cp -r temp_menagerie/franka_panda scenes/manipulation/
cp -r temp_menagerie/universal_robots_ur5e scenes/manipulation/
cp -r temp_menagerie/anybotics_anymal_c scenes/locomotion/

# Clean up
# rm -rf temp_menagerie

# 2. Get classic control tasks
# You'll need to extract these from gym/dm_control
echo "✓ Scenes ready in ./scenes/"
