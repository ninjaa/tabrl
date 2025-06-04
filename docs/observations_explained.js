// Understanding MuJoCo Observations

/**
 * WHAT ARE THESE OBSERVATIONS?
 * 
 * qpos = "Generalized Position" - WHERE things are
 * qvel = "Generalized Velocity" - HOW FAST things are moving  
 * sensordata = Extra sensors YOU defined in XML
 */

// Example 1: Simple Robot Arm (6 DOF)
const armObservations = {
    qpos: [
        0.1,   // Joint 1 angle (radians)
        -0.5,  // Joint 2 angle
        0.3,   // Joint 3 angle  
        0.0,   // Joint 4 angle
        0.7,   // Joint 5 angle
        0.2    // Joint 6 angle
    ],
    qvel: [
        0.0,   // Joint 1 angular velocity (rad/s)
        0.1,   // Joint 2 angular velocity
        -0.05, // Joint 3 angular velocity
        0.0,   // Joint 4 angular velocity
        0.0,   // Joint 5 angular velocity
        0.02   // Joint 6 angular velocity
    ],
    sensordata: [] // No extra sensors defined
};

// Example 2: Humanoid Robot
const humanoidObservations = {
    qpos: [
        // Root position (floating base)
        0.0, 1.2, 0.0,     // x, y, z position in world
        1.0, 0.0, 0.0, 0.0, // quaternion orientation (w,x,y,z)
        
        // Joint angles
        0.1,   // left_hip_pitch
        -0.2,  // left_knee
        0.05,  // left_ankle
        0.15,  // right_hip_pitch
        -0.25, // right_knee
        0.0,   // right_ankle
        // ... more joints for arms, torso, etc.
    ],
    qvel: [
        // Root velocity
        0.5, 0.0, 0.1,    // linear velocity (m/s)
        0.0, 0.1, 0.0,    // angular velocity (rad/s)
        
        // Joint velocities
        0.2,   // left_hip_pitch velocity
        -0.5,  // left_knee velocity
        // ... etc
    ],
    sensordata: [
        // If you defined sensors in XML:
        0.98,  // accelerometer_x
        0.02,  // accelerometer_y
        -9.81, // accelerometer_z
        15.2,  // force_sensor_left_foot
        18.7,  // force_sensor_right_foot
    ]
};

/**
 * BETTER OBSERVATION EXTRACTION
 * This is more realistic - you usually want specific, normalized observations
 */
function getSmartObservations(sim, sceneType) {
    const obs = [];
    
    if (sceneType === 'arm_manipulation') {
        // For a robot arm, you might want:
        
        // 1. Joint angles (normalized to [-1, 1])
        const jointLimits = [
            [-2.9, 2.9],   // joint1 limits
            [-1.76, 1.76], // joint2 limits
            // etc...
        ];
        
        for (let i = 0; i < 6; i++) {
            const angle = sim.data.qpos[i];
            const [min, max] = jointLimits[i];
            const normalized = 2 * (angle - min) / (max - min) - 1;
            obs.push(normalized);
        }
        
        // 2. Joint velocities (clipped)
        for (let i = 0; i < 6; i++) {
            const vel = Math.max(-5, Math.min(5, sim.data.qvel[i]));
            obs.push(vel / 5); // normalize to [-1, 1]
        }
        
        // 3. End effector position (from forward kinematics)
        const endEffectorPos = sim.data.site_xpos[0]; // assuming site at gripper
        obs.push(endEffectorPos[0]); // x
        obs.push(endEffectorPos[1]); // y
        obs.push(endEffectorPos[2]); // z
        
        // 4. Target object position (if doing pick and place)
        const targetPos = sim.data.body_xpos[5]; // assuming body 5 is target
        obs.push(targetPos[0]);
        obs.push(targetPos[1]); 
        obs.push(targetPos[2]);
        
        // 5. Relative position (target - end effector)
        obs.push(targetPos[0] - endEffectorPos[0]);
        obs.push(targetPos[1] - endEffectorPos[1]);
        obs.push(targetPos[2] - endEffectorPos[2]);
        
        // 6. Gripper state
        obs.push(sim.data.qpos[6]); // gripper opening
        
        // 7. Contact/touch sensors
        if (sim.data.sensordata.length > 0) {
            obs.push(sim.data.sensordata[0]); // gripper touch sensor
        }
        
    } else if (sceneType === 'humanoid_locomotion') {
        // For walking, you want different things:
        
        // 1. Body orientation (crucial for balance)
        const rootQuat = [
            sim.data.qpos[3], // w
            sim.data.qpos[4], // x  
            sim.data.qpos[5], // y
            sim.data.qpos[6]  // z
        ];
        // Convert to euler angles or rotation matrix features
        
        // 2. Center of mass velocity
        obs.push(sim.data.qvel[0]); // forward velocity (important!)
        obs.push(sim.data.qvel[1]); // lateral velocity
        obs.push(sim.data.qvel[2]); // vertical velocity
        
        // 3. Joint angles (hips, knees, ankles)
        for (let i = 7; i < sim.data.qpos.length; i++) {
            obs.push(sim.data.qpos[i]);
        }
        
        // 4. Foot contact sensors
        const leftFootContact = sim.data.sensordata[3] > 0.1 ? 1.0 : 0.0;
        const rightFootContact = sim.data.sensordata[4] > 0.1 ? 1.0 : 0.0;
        obs.push(leftFootContact);
        obs.push(rightFootContact);
        
        // 5. IMU data (if defined)
        if (sim.data.sensordata.length > 5) {
            obs.push(sim.data.sensordata[0]); // accel_x
            obs.push(sim.data.sensordata[1]); // accel_y
            obs.push(sim.data.sensordata[2]); // accel_z
        }
    }
    
    return new Float32Array(obs);
}

/**
 * ABOUT CAMERAS/VISION
 * 
 * MuJoCo cameras are NOT in sensordata by default!
 * If you want vision, you need to:
 */

async function getCameraObservations(sim, renderer) {
    // 1. Define camera in XML:
    // <camera name="robot_eye" pos="0 0 0.5" mode="fixed" fovy="60"/>
    
    // 2. Render from that camera
    const width = 84;  // Small images for RL
    const height = 84;
    const pixels = renderer.renderCamera(sim, 'robot_eye', width, height);
    
    // 3. Process pixels (RGB to grayscale, normalize, etc.)
    const processed = processImage(pixels, width, height);
    
    return processed;
}

/**
 * WHAT OBSERVATIONS TO USE?
 * 
 * It depends on your task!
 */

const OBSERVATION_STRATEGIES = {
    // Manipulation: Focus on proprioception + target
    manipulation: {
        includes: [
            'joint_positions',
            'joint_velocities', 
            'end_effector_pos',
            'target_object_pos',
            'relative_positions',
            'gripper_state',
            'touch_sensors'
        ],
        typical_dim: 25  // ~25 numbers
    },
    
    // Locomotion: Focus on balance + gait
    locomotion: {
        includes: [
            'body_orientation',
            'center_of_mass_velocity',
            'joint_angles',
            'joint_velocities',
            'foot_contacts',
            'IMU_readings'
        ],
        typical_dim: 40  // ~40 numbers
    },
    
    // Vision-based: Add camera
    vision_based: {
        includes: [
            'low_dim_state',  // Some proprioception
            'camera_rgb',     // 84x84x3 = 21,168 pixels!
            'depth_image'     // Optional depth
        ],
        typical_dim: 21000+  // Huge!
    }
};

/**
 * EXAMPLE: Complete observation function for TabRL
 */
export function getObservationsForTask(sim, taskType) {
    switch(taskType) {
        case 'pick_and_place':
            // Get arm joint info + object positions
            return getArmManipulationObs(sim);
            
        case 'walk_forward':
            // Get balance and gait info
            return getLocomotionObs(sim);
            
        case 'visual_sorting':
            // Get camera + basic state
            return getVisionObs(sim);
            
        default:
            // Fallback: just dump everything
            return getAllObs(sim);
    }
}

// Helper to check what's available
export function analyzeAvailableObservations(model) {
    console.log('Available observations:');
    console.log(`- Position dims (qpos): ${model.nq}`);
    console.log(`- Velocity dims (qvel): ${model.nv}`);
    console.log(`- Sensors: ${model.nsensor}`);
    console.log(`- Total standard obs: ${model.nq + model.nv + model.nsensor}`);
    
    // List sensor names if any
    if (model.sensor_names) {
        console.log('Sensor names:', model.sensor_names);
    }
}