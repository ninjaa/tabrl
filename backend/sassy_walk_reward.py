def reward_fn(state, action):
    # Dense reward for sassy hip-swinging walk
    
    # Forward progress reward
    forward_velocity = state.xd.vel[0, 0]  # x-velocity of root body
    forward_reward = jnp.clip(forward_velocity, 0.0, 2.0) * 2.0
    
    # Hip swaying reward - lateral movement of torso
    lateral_velocity = jnp.abs(state.xd.vel[0, 1])  # y-velocity magnitude
    hip_sway_reward = jnp.clip(lateral_velocity, 0.0, 0.5) * 3.0
    
    # Rhythmic hip motion - reward periodic lateral movement
    time_proxy = jnp.sum(state.q[:4]) * 0.1  # Use joint positions as time proxy
    target_sway = 0.3 * jnp.sin(time_proxy * 2.0)
    actual_sway = state.x.pos[0, 1]  # y-position of root
    rhythm_reward = 1.0 - jnp.abs(target_sway - actual_sway)
    
    # Upright stability bonus
    height = state.x.pos[0, 2]  # z-position (height)
    upright_reward = jnp.clip(height, 0.2, 0.8) * 1.5
    
    # Energy efficiency penalty
    energy_penalty = -0.001 * jnp.sum(action**2)
    
    # Combine rewards
    total_reward = forward_reward + hip_sway_reward + rhythm_reward + upright_reward + energy_penalty
    
    return float(total_reward)
