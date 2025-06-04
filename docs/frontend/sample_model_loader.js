// scene_loader.js - Clean scene loading for TabRL

/**
 * Load a MuJoCo scene from a directory containing XML and assets
 * @param {string} scenePath - Path to scene directory (e.g., "/scenes/panda_arm")
 * @returns {Object} Loaded model and simulation instances
 */
async function loadScene(scenePath) {
    try {
        // 1. Load the main XML file
        const xmlResponse = await fetch(`${scenePath}/scene.xml`);
        if (!xmlResponse.ok) {
            throw new Error(`Failed to load scene XML: ${xmlResponse.status}`);
        }
        const xmlText = await xmlResponse.text();
        
        // 2. Create asset loader for meshes and textures
        const assetLoader = async (path) => {
            console.log(`Loading asset: ${path}`);
            
            const assetResponse = await fetch(`${scenePath}/${path}`);
            if (!assetResponse.ok) {
                console.warn(`Failed to load asset: ${path}`);
                return null;
            }
            
            return await assetResponse.arrayBuffer();
        };
        
        // 3. Load the model with MuJoCo WASM
        const model = await mujoco.loadXML(xmlText, { assetLoader });
        
        // 4. Create simulation instance
        const sim = new mujoco.Simulation(model);
        
        // 5. Extract useful information
        const sceneInfo = {
            name: model.name,
            
            // Dimensions
            nq: model.nq,          // Number of joint positions
            nv: model.nv,          // Number of joint velocities  
            nu: model.nu,          // Number of actuators (action dimension)
            nsensor: model.nsensor, // Number of sensors
            
            // Key data pointers
            qpos: sim.data.qpos,       // Joint positions
            qvel: sim.data.qvel,       // Joint velocities
            ctrl: sim.data.ctrl,       // Motor commands (what policy outputs)
            sensordata: sim.data.sensordata, // Sensor readings
            
            // Useful for rendering
            camera: model.camera,
            light: model.light
        };
        
        console.log('Scene loaded successfully:', sceneInfo);
        
        return {
            model,
            sim,
            info: sceneInfo
        };
        
    } catch (error) {
        console.error('Failed to load scene:', error);
        throw error;
    }
}

/**
 * Initialize and run a simulation loop
 * @param {Object} scene - Loaded scene from loadScene()
 * @param {Function} policyFunction - Function that maps observations to actions
 */
async function runSimulation(scene, policyFunction) {
    const { sim, info } = scene;
    
    // Reset simulation to initial state
    sim.reset();
    
    // Simulation loop
    const runLoop = async () => {
        // 1. Get current observation
        const observation = getObservation(sim);
        
        // 2. Query policy for action
        const action = await policyFunction(observation);
        
        // 3. Apply action to motors
        for (let i = 0; i < info.nu; i++) {
            sim.data.ctrl[i] = action[i];
        }
        
        // 4. Step physics forward
        sim.step();
        
        // 5. Render (if renderer attached)
        if (window.mujocoRenderer) {
            window.mujocoRenderer.render(sim);
        }
        
        // Continue loop
        requestAnimationFrame(runLoop);
    };
    
    // Start the loop
    runLoop();
}

/**
 * Extract observation vector from simulation state
 * @param {Object} sim - MuJoCo simulation instance
 * @returns {Float32Array} Observation vector for policy
 */
function getObservation(sim) {
    // Combine different observation sources
    const observations = [];
    
    // Joint positions
    for (let i = 0; i < sim.data.qpos.length; i++) {
        observations.push(sim.data.qpos[i]);
    }
    
    // Joint velocities
    for (let i = 0; i < sim.data.qvel.length; i++) {
        observations.push(sim.data.qvel[i]);
    }
    
    // Sensor data (if any)
    for (let i = 0; i < sim.data.sensordata.length; i++) {
        observations.push(sim.data.sensordata[i]);
    }
    
    return new Float32Array(observations);
}

/**
 * Example usage in TabRL
 */
async function initializeTabRL() {
    // Load a scene
    const scene = await loadScene('/scenes/panda_tabletop');
    
    // Create a simple policy (later this comes from ONNX)
    const dummyPolicy = async (observation) => {
        // For now, just return zeros (no movement)
        return new Float32Array(scene.info.nu).fill(0);
    };
    
    // Run the simulation
    await runSimulation(scene, dummyPolicy);
}

// Scene catalog helper
const SCENE_CATALOG = {
    'panda_tabletop': {
        path: '/scenes/manipulation/panda_tabletop',
        description: 'Panda arm with blocks on table',
        difficulty: 'medium',
        taskType: 'manipulation'
    },
    'cartpole': {
        path: '/scenes/control/cartpole',
        description: 'Classic cart-pole balancing',
        difficulty: 'easy',
        taskType: 'control'
    },
    'humanoid_walk': {
        path: '/scenes/locomotion/humanoid',
        description: 'Humanoid robot walking',
        difficulty: 'hard',
        taskType: 'locomotion'
    }
};

/**
 * Load scene from catalog
 */
async function loadSceneFromCatalog(sceneName) {
    const sceneConfig = SCENE_CATALOG[sceneName];
    if (!sceneConfig) {
        throw new Error(`Unknown scene: ${sceneName}`);
    }
    
    console.log(`Loading ${sceneConfig.description}...`);
    return await loadScene(sceneConfig.path);
}

export { 
    loadScene, 
    runSimulation, 
    getObservation, 
    loadSceneFromCatalog,
    SCENE_CATALOG 
};