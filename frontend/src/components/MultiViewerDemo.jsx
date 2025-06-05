import React, { useState } from 'react';
import PlaygroundViewer from './PlaygroundViewer';

const MultiViewerDemo = () => {
  const [viewers, setViewers] = useState([
    { id: 'viewer-1', width: 600, height: 400 },
    { id: 'viewer-2', width: 600, height: 400 }
  ]);
  
  const [inferenceLog, setInferenceLog] = useState([]);

  const addViewer = () => {
    const newId = `viewer-${viewers.length + 1}`;
    setViewers([...viewers, { id: newId, width: 600, height: 400 }]);
  };

  const removeViewer = (viewerId) => {
    setViewers(viewers.filter(v => v.id !== viewerId));
  };

  const handleSceneLoad = (viewerId, sceneInfo) => {
    console.log(`Viewer ${viewerId} loaded scene:`, sceneInfo);
    addToLog(`[${viewerId}] Scene loaded: ${sceneInfo.sceneId}`);
  };

  const handleInferenceUpdate = (viewerId, data) => {
    addToLog(`[${viewerId}] Inference: ${data.actions?.slice(0, 3).map(a => a.toFixed(2)).join(', ')}...`);
  };

  const addToLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setInferenceLog(prev => [...prev.slice(-19), `${timestamp}: ${message}`]);
  };

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <h2>Multiple MuJoCo Playground Viewers</h2>
        <p>Each viewer can load different scenes and run independent inference from the backend.</p>
        
        <div style={{ marginBottom: '16px' }}>
          <button 
            onClick={addViewer}
            style={{ 
              padding: '8px 16px', 
              backgroundColor: '#4CAF50', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              marginRight: '8px'
            }}
          >
            Add Viewer
          </button>
          
          <span style={{ color: '#666' }}>
            Total viewers: {viewers.length}
          </span>
        </div>
      </div>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(650px, 1fr))', 
        gap: '20px',
        marginBottom: '20px'
      }}>
        {viewers.map((viewer) => (
          <div key={viewer.id} style={{ position: 'relative' }}>
            <div style={{ 
              position: 'absolute', 
              top: '-10px', 
              right: '-10px', 
              zIndex: 10 
            }}>
              <button
                onClick={() => removeViewer(viewer.id)}
                style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: '#f44336',
                  color: 'white',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
                title="Remove viewer"
              >
                ×
              </button>
            </div>
            
            <PlaygroundViewer
              instanceId={viewer.id}
              width={viewer.width}
              height={viewer.height}
              onSceneLoad={(sceneInfo) => handleSceneLoad(viewer.id, sceneInfo)}
              onInferenceUpdate={(data) => handleInferenceUpdate(viewer.id, data)}
            />
          </div>
        ))}
      </div>

      {/* Inference Log */}
      <div style={{ 
        border: '1px solid #ddd', 
        borderRadius: '8px', 
        padding: '16px',
        backgroundColor: '#f9f9f9'
      }}>
        <h3 style={{ margin: '0 0 12px 0' }}>Inference Log</h3>
        <div style={{ 
          height: '200px', 
          overflowY: 'auto', 
          fontFamily: 'monospace', 
          fontSize: '12px',
          backgroundColor: '#000',
          color: '#00ff00',
          padding: '8px',
          borderRadius: '4px'
        }}>
          {inferenceLog.length === 0 ? (
            <div style={{ color: '#666' }}>No inference activity yet...</div>
          ) : (
            inferenceLog.map((log, index) => (
              <div key={index}>{log}</div>
            ))
          )}
        </div>
      </div>

      {/* Usage Instructions */}
      <div style={{ 
        marginTop: '20px', 
        padding: '16px', 
        backgroundColor: '#e3f2fd', 
        borderRadius: '8px' 
      }}>
        <h4>How to use:</h4>
        <ol>
          <li><strong>Select a scene:</strong> Choose from locomotion, manipulation, or dm_control_suite environments</li>
          <li><strong>Load scene:</strong> Click "Load Scene" to download and render the robot in MuJoCo WASM</li>
          <li><strong>Start inference:</strong> Click "Start Inference" to connect to backend JAX policy</li>
          <li><strong>Multiple viewers:</strong> Add more viewers to compare different scenes or policies</li>
        </ol>
        
        <p><strong>Note:</strong> Make sure your backend is running at <code>localhost:8000</code> with the playground and inference endpoints available.</p>
      </div>
    </div>
  );
};

export default MultiViewerDemo;
