import React, { useState, useEffect } from 'react';
import { testWebContainer, testSettings } from './webcontainer-test';

function App() {
  const [status, setStatus] = useState('Initializing...');
  const [webContainerReady, setWebContainerReady] = useState(false);
  const [keys, setKeys] = useState({
    anthropic_api_key: '',
    modal_token_id: '',
    modal_token_secret: ''
  });
  
  useEffect(() => {
    // Boot WebContainer without keys initially
    testWebContainer().then(() => {
      setStatus('WebContainer ready! Paste your API keys below.');
      setWebContainerReady(true);
    }).catch((err) => {
      setStatus(`WebContainer failed: ${err.message}`);
    });
  }, []);
  
  const handleSaveKeys = async () => {
    try {
      const response = await fetch('http://localhost:3000/settings/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(keys)
      });
      const result = await response.json();
      setStatus(`Keys saved securely in WebContainer! Keys: ${result.keys_stored?.join(', ')}`);
    } catch (err) {
      setStatus(`Failed to save keys: ${err.message}`);
    }
  };
  
  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>TabRL - Setup Test</h1>
      <p>Status: {status}</p>
      
      <h2>Configure API Keys</h2>
      <div style={{ marginBottom: '10px' }}>
        <input 
          placeholder="Anthropic API Key (sk-ant-...)"
          value={keys.anthropic_api_key}
          onChange={(e) => setKeys({...keys, anthropic_api_key: e.target.value})}
          style={{ width: '400px', marginRight: '10px' }}
          type="password"
        />
      </div>
      <div style={{ marginBottom: '10px' }}>
        <input 
          placeholder="Modal Token ID (ak-...)"  
          value={keys.modal_token_id}
          onChange={(e) => setKeys({...keys, modal_token_id: e.target.value})}
          style={{ width: '400px', marginRight: '10px' }}
          type="password"
        />
      </div>
      <div style={{ marginBottom: '10px' }}>
        <input 
          placeholder="Modal Token Secret (as-...)"  
          value={keys.modal_token_secret}
          onChange={(e) => setKeys({...keys, modal_token_secret: e.target.value})}
          style={{ width: '400px', marginRight: '10px' }}
          type="password"
        />
      </div>
      <button onClick={handleSaveKeys} disabled={!webContainerReady}>
        Save Keys Securely in WebContainer
      </button>
      
      {webContainerReady && (
        <>
          <h2>Test Modal Connection</h2>
          <button onClick={async () => {
            const resp = await fetch('http://localhost:3000/test/modal', {
              method: 'POST'
            });
            const data = await resp.json();
            setStatus(`Modal test: ${JSON.stringify(data)}`);
          }}>Test Modal</button>
          
          <h2>WebContainer Status</h2>
          <button onClick={async () => {
            const resp = await fetch('http://localhost:3000/settings/load');
            const data = await resp.json();
            setStatus(`Keys in WebContainer: ${JSON.stringify(data)}`);
          }}>Check Stored Keys</button>
        </>
      )}
      
      <h2>Setup Progress</h2>
      <ul>
        <li>✓ WebContainer boots</li>
        <li>◯ API keys stored securely in WebContainer</li>
        <li>◯ Modal connection test</li>
        <li>◯ Load MuJoCo scene</li>
        <li>◯ Run ONNX inference</li>
      </ul>
    </div>
  );
}

export default App;
