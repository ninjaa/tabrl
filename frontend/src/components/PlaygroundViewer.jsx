import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const PlaygroundViewer = ({ 
  instanceId = 'default',
  width = 800, 
  height = 600,
  onSceneLoad = () => {},
  onInferenceUpdate = () => {} 
}) => {
  const [scenes, setScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState('');
  const [sceneLoading, setSceneLoading] = useState(false);
  const [status, setStatus] = useState('Initializing...');
  const [mujoco, setMujoco] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [inferenceRunning, setInferenceRunning] = useState(false);
  
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const animationRef = useRef(null);
  const websocketRef = useRef(null);
  const controlsRef = useRef(null);
  const mujocoRef = useRef(null);
  const modelRef = useRef(null);
  const stateRef = useRef(null);
  const simulationRef = useRef(null);

  // Fetch available playground scenes
  useEffect(() => {
    const fetchScenes = async () => {
      try {
        console.log('Fetching scenes from:', 'http://localhost:8000/api/playground/environments');
        const response = await fetch('http://localhost:8000/api/playground/environments');
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        // Flatten all categories into a single list
        const allScenes = [];
        Object.entries(data).forEach(([category, envs]) => {
          envs.forEach(env => {
            allScenes.push({
              id: `${category}/${env}`,
              name: env,
              category: category,
              displayName: `${category}: ${env}`
            });
          });
        });
        
        setScenes(allScenes);
        setStatus('Ready to load scene');
      } catch (error) {
        console.error('Failed to fetch scenes:', error);
        setStatus(`Failed to load scenes: ${error.message}`);
      }
    };

    fetchScenes();
  }, []);

  // Initialize MuJoCo WASM
  useEffect(() => {
    const initMujoco = async () => {
      try {
        setStatus('Loading MuJoCo WASM...');
        
        // Use the same approach as the working MujocoViewer
        const mujocoModule = await import('/mujoco_wasm/dist/mujoco_wasm.js');
        
        setStatus('Initializing MuJoCo...');
        
        // The module exports a default function that returns the MuJoCo instance
        const loadMujoco = mujocoModule.default;
        const mj = await loadMujoco();
        
        mujocoRef.current = mj;
        setMujoco(mj);
        
        console.log('MuJoCo WASM loaded successfully');
        
        // Log available MuJoCo methods/properties for debugging
        console.log('MuJoCo methods:', Object.keys(mj).filter(k => typeof mj[k] === 'function').slice(0, 20));
        console.log('MuJoCo properties:', Object.keys(mj).filter(k => typeof mj[k] !== 'function').slice(0, 20));
        
        // Set up Emscripten's Virtual File System (following MuJoCo examples)
        try {
          // Create /working directory if it doesn't exist
          if (!mj.FS.analyzePath('/working').exists) {
            mj.FS.mkdir('/working');
          }
          
          // Mount MEMFS to /working for better performance
          try {
            mj.FS.mount(mj.MEMFS, { root: '.' }, '/working');
          } catch (e) {
            // Might already be mounted, that's okay
            console.log('MEMFS mount skipped (possibly already mounted)');
          }
        } catch (e) {
          console.error('Error setting up filesystem:', e);
        }
        
        // Clean up any existing files in /working
        try {
          const files = mj.FS.readdir('/working');
          files.forEach(file => {
            if (file !== '.' && file !== '..') {
              try {
                mj.FS.unlink(`/working/${file}`);
              } catch (e) {
                // Ignore errors
              }
            }
          });
        } catch (e) {
          // Directory might not exist
        }
        
        setStatus('MuJoCo loaded, select a scene');
      } catch (error) {
        console.error('Failed to load MuJoCo:', error);
        setStatus(`Failed to load MuJoCo: ${error.message}`);
      }
    };

    initMujoco();
  }, []);

  // Initialize Three.js scene
  const initThreeScene = () => {
    if (!containerRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    scene.fog = new THREE.Fog(0xf0f0f0, 10, 50);

    // Camera
    const camera = new THREE.PerspectiveCamera(
      45,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      100
    );
    camera.position.set(3, 3, 5);
    camera.lookAt(0, 0, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.left = -10;
    directionalLight.shadow.camera.right = 10;
    directionalLight.shadow.camera.top = 10;
    directionalLight.shadow.camera.bottom = -10;
    directionalLight.shadow.camera.near = 0.1;
    directionalLight.shadow.camera.far = 50;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(20, 20);
    const groundMaterial = new THREE.MeshLambertMaterial({ color: 0xcccccc });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Grid helper
    const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0xdddddd);
    scene.add(gridHelper);

    // Store references
    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;

    // Add orbit controls for interactivity
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);
    controls.update();
    controlsRef.current = controls;

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  };

  useEffect(initThreeScene, []);

  // Load selected scene
  const loadScene = async (sceneId) => {
    if (!mujocoRef.current || !sceneId) return;

    setSceneLoading(true);
    setStatus('Loading scene...');

    try {
      // SceneId is in format "category/sceneName", so parse it
      const [category, sceneName] = sceneId.split('/');
      
      // Fetch complete scene package with resolved XML
      const response = await fetch(`http://localhost:8000/api/playground/${category}/${sceneName}/scene-package`);
      if (!response.ok) {
        throw new Error(`Failed to load scene: ${response.status}`);
      }
      
      const packageData = await response.json();
      const sceneXML = packageData.xml;
      const assets = packageData.assets || {};
      
      if (!sceneXML) {
        throw new Error('No XML content received');
      }
      
      const mujoco = mujocoRef.current;
      
      console.log('Loading scene with MuJoCo:', mujoco);
      
      // Clean up previous model
      if (simulationRef.current) {
        simulationRef.current.free();
        simulationRef.current = null;
      }
      if (stateRef.current) {
        stateRef.current.free();
        stateRef.current = null;
      }
      if (modelRef.current) {
        modelRef.current.free();
        modelRef.current = null;
      }
      
      // Clear previous files from virtual filesystem
      try {
        const files = mujocoRef.current.FS.readdir('/working');
        files.forEach(file => {
          if (file !== '.' && file !== '..') {
            try {
              mujocoRef.current.FS.unlink(`/working/${file}`);
            } catch (e) {
              // Ignore errors for directories
            }
          }
        });
      } catch (e) {
        console.warn('Could not clear /working directory:', e);
      }
      
      // Load assets
      if (assets && Object.keys(assets).length > 0) {
        console.log(`Loading ${Object.keys(assets).length} assets...`);
        
        for (const [filename, content] of Object.entries(assets)) {
          try {
            // Decode base64 content
            const binaryString = atob(content);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
              bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Parse the XML to find meshdir
            let meshdir = '';
            try {
              const parser = new DOMParser();
              const xmlDoc = parser.parseFromString(sceneXML, 'text/xml');
              const compilerElement = xmlDoc.querySelector('compiler');
              if (compilerElement && compilerElement.getAttribute('meshdir')) {
                meshdir = compilerElement.getAttribute('meshdir');
              }
            } catch (e) {
              console.warn('Could not parse meshdir from XML:', e);
            }
            
            // Write to both locations for maximum compatibility
            // 1. Write directly to /working/ (some scenes might expect this)
            const directPath = `/working/${filename}`;
            mujocoRef.current.FS.writeFile(directPath, bytes);
            console.log(`Loaded asset: ${directPath}`);
            
            // 2. Also write to meshdir if specified (following MuJoCo examples pattern)
            if (meshdir) {
              const meshdirPath = `/working/${meshdir}`;
              if (!mujocoRef.current.FS.analyzePath(meshdirPath).exists) {
                mujocoRef.current.FS.mkdir(meshdirPath);
              }
              const fullPath = `${meshdirPath}/${filename}`;
              mujocoRef.current.FS.writeFile(fullPath, bytes);
              console.log(`Loaded asset: ${fullPath}`);
            }
          } catch (error) {
            console.error(`Failed to load asset ${filename}:`, error);
          }
        }
      }
      
      // Write the main scene XML
      mujocoRef.current.FS.writeFile('/working/scene.xml', sceneXML);
      
      // List files in /working to verify everything is loaded
      console.log('Files in /working:', mujocoRef.current.FS.readdir('/working').filter(f => f !== '.' && f !== '..'));
      if (meshdir) {
        try {
          const meshdirFiles = mujocoRef.current.FS.readdir(`/working/${meshdir}`);
          console.log(`Files in /working/${meshdir}:`, meshdirFiles.filter(f => f !== '.' && f !== '..'));
        } catch (e) {
          console.warn(`Could not list files in /working/${meshdir}`);
        }
      }
      
      // Create model, state, and simulation
      console.log('Creating MuJoCo model from scene XML...');
      try {
        // First try a minimal test to ensure MuJoCo is working
        const minimalXML = `<mujoco>
          <worldbody>
            <body name="test">
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>`;
        
        try {
          mujocoRef.current.FS.writeFile('/working/minimal.xml', minimalXML);
          const testModel = new mujoco.Model('/working/minimal.xml');
          console.log('✓ MuJoCo can load minimal XML files');
          testModel.free();
        } catch (e) {
          console.error('✗ MuJoCo cannot even load minimal XML:', e);
        }
        
        // Try to create the model
        const testXML = `<mujoco>
          <worldbody>
            <body>
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>`;
        
        try {
          mujocoRef.current.FS.writeFile('/working/test.xml', testXML);
          const testModel = new mujoco.Model('/working/test.xml');
          console.log('✓ MuJoCo WASM is working correctly with simple scenes');
          // Clean up test
          testModel.free();
        } catch (testError) {
          console.error('✗ MuJoCo WASM failed even with simple test scene:', testError);
        }
        
        // Save the XML for debugging
        console.log('XML content preview (first 500 chars):', sceneXML.substring(0, 500));
        
        // Try to create the model
        modelRef.current = new mujoco.Model('/working/scene.xml');
        console.log('Model created successfully');
      } catch (modelError) {
        console.error('Model creation error details:', modelError);
        console.error('Error name:', modelError.name);
        console.error('Error message:', modelError.message);
        console.error('Error stack:', modelError.stack);
        
        // Try to get MuJoCo-specific error information
        try {
          // Check if MuJoCo has an error buffer we can read
          if (mujocoRef.current._mj_error && mujocoRef.current._mj_error_msg) {
            const errorCode = mujocoRef.current._mj_error();
            console.error('MuJoCo error code:', errorCode);
            
            // Try to get error message pointer
            const errorMsgPtr = mujocoRef.current._mj_error_msg();
            if (errorMsgPtr) {
              const errorMsg = mujocoRef.current.UTF8ToString(errorMsgPtr);
              console.error('MuJoCo error message:', errorMsg);
              setStatus(`MuJoCo error: ${errorMsg}`);
            }
          }
          
          // Try to check warning buffer as well
          if (mujocoRef.current._mj_warning && mujocoRef.current._mj_warning_msg) {
            const warningCode = mujocoRef.current._mj_warning();
            if (warningCode) {
              const warningMsgPtr = mujocoRef.current._mj_warning_msg();
              if (warningMsgPtr) {
                const warningMsg = mujocoRef.current.UTF8ToString(warningMsgPtr);
                console.warn('MuJoCo warning:', warningMsg);
              }
            }
          }
          
          console.error('MuJoCo error info:', modelError.toString());
        } catch (e) {
          console.error('Could not get detailed error info:', e);
        }
        
        throw modelError;
      }
      
      stateRef.current = new mujoco.State(modelRef.current);
      simulationRef.current = new mujoco.Simulation(modelRef.current, stateRef.current);
      
      setSimulation(simulationRef.current);
      
      // Clear existing scene objects (except lights and ground)
      const objectsToRemove = [];
      sceneRef.current.traverse((child) => {
        if (child.isMesh && child.material?.color?.getHex() !== 0xcccccc) {
          objectsToRemove.push(child);
        }
      });
      objectsToRemove.forEach(obj => sceneRef.current.remove(obj));

      // Create visual representation from MuJoCo model
      await createVisualFromModel(modelRef.current, simulationRef.current);
      
      // Start animation loop
      startAnimation();
      
      setStatus(`Scene loaded: ${sceneId}`);
      onSceneLoad({ sceneId, model: modelRef.current, simulation: simulationRef.current });
      
    } catch (error) {
      console.error('Failed to load scene:', error);
      setStatus(`Failed to load scene: ${error.message}`);
    } finally {
      setSceneLoading(false);
    }
  };

  // Create visual representation from MuJoCo model
  const createVisualFromModel = async (model, simulation) => {
    if (!model || !simulation) return;
    
    // For now, create a simple placeholder for each body
    const nbody = model.nbody();
    
    for (let i = 1; i < nbody; i++) { // Skip world body (0)
      const bodyName = model.id2name(i, model.mjOBJ_BODY);
      
      // Get body position from simulation data
      const bodyId = model.name2id(model.mjOBJ_BODY, bodyName);
      if (bodyId >= 0) {
        // Create a simple box placeholder
        const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
        const material = new THREE.MeshStandardMaterial({ 
          color: 0x0080ff,
          metalness: 0.4,
          roughness: 0.6
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `body_${bodyName}`;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        sceneRef.current.add(mesh);
      }
    }
    
    console.log(`Created ${nbody - 1} visual bodies`);
  };

  // Animation loop
  const startAnimation = () => {
    if (animationRef.current) return; // Already running

    const animate = () => {
      // Check if all required refs exist
      if (!sceneRef.current || !cameraRef.current || !rendererRef.current) {
        animationRef.current = null;
        return;
      }

      // Update simulation if available
      if (simulationRef.current && !inferenceRunning) {
        simulationRef.current.step();
        
        // Update visual representation based on simulation state
        updateVisualsFromSimulation(simulationRef.current);
      }

      // Update controls if available
      if (controlsRef.current) {
        controlsRef.current.update();
      }

      // Render
      rendererRef.current.render(sceneRef.current, cameraRef.current);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();
  };

  // Update Three.js objects based on simulation state
  const updateVisualsFromSimulation = (simulation) => {
    if (!simulation || !sceneRef.current) return;

    // Update robot positions based on simulation state
    sceneRef.current.traverse((child) => {
      if (child.userData.isRobot) {
        try {
          // Get simulation data safely
          const qpos = simulation.data.qpos;
          if (qpos && qpos.length >= 3) {
            child.position.x = qpos[0];
            child.position.y = qpos[1]; 
            child.position.z = Math.max(qpos[2], 0.5); // Keep above ground
          }
        } catch (e) {
          // If we can't read simulation state, just animate
          child.rotation.y += 0.01;
        }
      }
    });
  };

  // Start inference connection
  const startInference = async () => {
    if (!selectedScene || !simulationRef.current) return;

    try {
      setInferenceRunning(true);
      setStatus('Connecting to inference...');

      // Connect to backend inference WebSocket
      const wsUrl = `ws://127.0.0.1:8000/api/models/inference?scene=${selectedScene}&instance=${instanceId}`;
      websocketRef.current = new WebSocket(wsUrl);

      websocketRef.current.onopen = () => {
        setStatus('Inference active');
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'action') {
            // Apply action to simulation
            const actions = data.actions;
            for (let i = 0; i < actions.length && i < simulationRef.current.data.ctrl.length; i++) {
              simulationRef.current.data.ctrl[i] = actions[i];
            }
            
            onInferenceUpdate(data);
          }
        } catch (error) {
          console.error('Failed to parse inference data:', error);
        }
      };

      websocketRef.current.onerror = () => {
        setStatus('Inference connection failed');
        setInferenceRunning(false);
      };

      websocketRef.current.onclose = () => {
        setStatus('Inference disconnected');
        setInferenceRunning(false);
      };

    } catch (error) {
      console.error('Failed to start inference:', error);
      setStatus('Failed to start inference');
      setInferenceRunning(false);
    }
  };

  // Stop inference
  const stopInference = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
    }
    setInferenceRunning(false);
    setStatus(`Scene loaded: ${selectedScene}`);
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  return (
    <div className="playground-viewer" style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '16px' }}>
      <div className="controls" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '8px' }}>
          <select 
            value={selectedScene} 
            onChange={(e) => setSelectedScene(e.target.value)}
            disabled={sceneLoading}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
          >
            <option value="">Select a scene...</option>
            {scenes.map(scene => (
              <option key={scene.id} value={scene.id}>
                {scene.displayName}
              </option>
            ))}
          </select>
          
          <button 
            onClick={() => loadScene(selectedScene)}
            disabled={!selectedScene || sceneLoading || !mujocoRef.current}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: '#4CAF50', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {sceneLoading ? 'Loading...' : 'Load Scene'}
          </button>

          <button
            onClick={inferenceRunning ? stopInference : startInference}
            disabled={!simulationRef.current}
            style={{
              padding: '8px 16px',
              backgroundColor: inferenceRunning ? '#f44336' : '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {inferenceRunning ? 'Stop Inference' : 'Start Inference'}
          </button>
        </div>
        
        <div style={{ fontSize: '14px', color: '#666' }}>
          Status: {status} | Instance: {instanceId}
        </div>
      </div>

      <div 
        ref={containerRef} 
        style={{ 
          width: `${width}px`, 
          height: `${height}px`,
          border: '1px solid #eee',
          borderRadius: '4px',
          overflow: 'hidden'
        }}
      />
    </div>
  );
};

export default PlaygroundViewer;
