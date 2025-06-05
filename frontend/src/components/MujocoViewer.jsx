import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const MujocoViewer = () => {
  const containerRef = useRef(null);
  const [mujoco, setMujoco] = useState(null);
  const [status, setStatus] = useState('Loading MuJoCo...');
  const [demo, setDemo] = useState(null);

  useEffect(() => {
    let mounted = true;

    const initMujoco = async () => {
      try {
        setStatus('Loading MuJoCo WASM module...');
        
        // Directly import the MuJoCo module
        const mujocoModule = await import('/mujoco_wasm/dist/mujoco_wasm.js');
        
        setStatus('Initializing MuJoCo...');
        
        // The module exports a default function that returns the MuJoCo instance
        const loadMujoco = mujocoModule.default;
        const mj = await loadMujoco();
        
        if (!mounted) return;
        
        setMujoco(mj);
        setStatus('Setting up 3D scene...');
        
        // Create a simple demo scene
        const demoInstance = new MujocoDemo(mj, containerRef.current);
        await demoInstance.init();
        
        setDemo(demoInstance);
        setStatus('Ready! Use mouse to rotate, zoom, and pan.');
        
      } catch (error) {
        console.error('Failed to load MuJoCo:', error);
        setStatus(`Error: ${error.message}`);
      }
    };

    initMujoco();

    return () => {
      mounted = false;
      if (demo) {
        demo.destroy();
      }
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '500px', position: 'relative' }}>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
          border: '2px solid #ddd',
          borderRadius: '8px',
          backgroundColor: '#f0f0f0'
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          padding: '8px 12px',
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          borderRadius: '4px',
          fontSize: '12px'
        }}
      >
        {status}
      </div>
    </div>
  );
};

// Simplified MuJoCo Demo class
class MujocoDemo {
  constructor(mujocoModule, container) {
    this.mujoco = mujocoModule;
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.animationId = null;
  }

  async init() {
    // Setup Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x87CEEB); // Sky blue

    // Setup camera
    const containerRect = this.container.getBoundingClientRect();
    this.camera = new THREE.PerspectiveCamera(
      45,
      containerRect.width / containerRect.height,
      0.001,
      100
    );
    this.camera.position.set(3, 2, 3);

    // Setup renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(containerRect.width, containerRect.height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Setup controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1, 0);
    this.controls.enableDamping = true;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    this.scene.add(directionalLight);

    // Add a simple ground plane
    const groundGeometry = new THREE.PlaneGeometry(20, 20);
    const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x808080 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Add a simple robot-like object as placeholder
    await this.createSimpleRobot();

    // Start animation loop
    this.animate();

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
  }

  async createSimpleRobot() {
    // Create a simple robot structure while we set up MuJoCo integration
    const robotGroup = new THREE.Group();
    robotGroup.name = 'SimpleRobot';

    // Body
    const bodyGeometry = new THREE.BoxGeometry(0.5, 0.8, 0.3);
    const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x4444ff });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 1.2;
    body.castShadow = true;
    robotGroup.add(body);

    // Head
    const headGeometry = new THREE.SphereGeometry(0.15);
    const headMaterial = new THREE.MeshPhongMaterial({ color: 0x6666ff });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 1.8;
    head.castShadow = true;
    robotGroup.add(head);

    // Arms
    for (let i = 0; i < 2; i++) {
      const armGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.6);
      const armMaterial = new THREE.MeshPhongMaterial({ color: 0x888888 });
      const arm = new THREE.Mesh(armGeometry, armMaterial);
      arm.position.set(i === 0 ? -0.35 : 0.35, 1.2, 0);
      arm.castShadow = true;
      robotGroup.add(arm);
    }

    // Legs
    for (let i = 0; i < 2; i++) {
      const legGeometry = new THREE.CylinderGeometry(0.08, 0.08, 0.8);
      const legMaterial = new THREE.MeshPhongMaterial({ color: 0x888888 });
      const leg = new THREE.Mesh(legGeometry, legMaterial);
      leg.position.set(i === 0 ? -0.15 : 0.15, 0.4, 0);
      leg.castShadow = true;
      robotGroup.add(leg);
    }

    this.scene.add(robotGroup);

    // Add some animation
    this.robotGroup = robotGroup;
  }

  animate() {
    this.animationId = requestAnimationFrame(() => this.animate());

    // Simple animation - make robot sway
    if (this.robotGroup) {
      this.robotGroup.rotation.y = Math.sin(Date.now() * 0.001) * 0.2;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  onWindowResize() {
    if (!this.container) return;
    
    const containerRect = this.container.getBoundingClientRect();
    this.camera.aspect = containerRect.width / containerRect.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(containerRect.width, containerRect.height);
  }

  destroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    if (this.renderer && this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
    window.removeEventListener('resize', () => this.onWindowResize());
  }
}

export default MujocoViewer;
