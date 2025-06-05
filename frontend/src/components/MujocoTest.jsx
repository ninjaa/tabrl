import React, { useEffect, useState } from 'react';

const MujocoTest = () => {
  const [status, setStatus] = useState('Checking MuJoCo WASM files...');

  useEffect(() => {
    const checkMujocoFiles = async () => {
      try {
        // Test if we can access the main files
        const jsResponse = await fetch('/mujoco_wasm/dist/mujoco_wasm.js');
        const wasmResponse = await fetch('/mujoco_wasm/dist/mujoco_wasm.wasm');
        
        if (jsResponse.ok && wasmResponse.ok) {
          setStatus('✅ MuJoCo WASM files are accessible!');
          
          // Try to load the actual MuJoCo module
          try {
            const script = document.createElement('script');
            script.src = '/mujoco_wasm/dist/mujoco_wasm.js';
            script.onload = () => {
              setStatus('✅ MuJoCo WASM files loaded and accessible!');
            };
            script.onerror = () => {
              setStatus('⚠️ Files accessible but failed to load MuJoCo module');
            };
            document.head.appendChild(script);
          } catch (error) {
            setStatus(`⚠️ Files accessible but error loading: ${error.message}`);
          }
        } else {
          setStatus('❌ Cannot access MuJoCo WASM files');
        }
      } catch (error) {
        setStatus(`❌ Error: ${error.message}`);
      }
    };

    checkMujocoFiles();
  }, []);

  return (
    <div style={{ 
      padding: '20px', 
      border: '2px solid #ddd', 
      borderRadius: '8px', 
      margin: '20px',
      backgroundColor: '#f9f9f9'
    }}>
      <h3>MuJoCo WASM Status</h3>
      <p>{status}</p>
      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        <p>Expected files:</p>
        <ul>
          <li>/mujoco_wasm/dist/mujoco_wasm.js</li>
          <li>/mujoco_wasm/dist/mujoco_wasm.wasm</li>
          <li>/mujoco_wasm/dist/mujoco_wasm.d.ts</li>
        </ul>
      </div>
    </div>
  );
};

export default MujocoTest;
