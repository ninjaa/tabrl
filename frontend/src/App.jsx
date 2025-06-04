import React, { useState, useEffect } from 'react';
import { testWebContainer, testSettings } from './webcontainer-test';

function App() {
  const [status, setStatus] = useState('Initializing...');
  const [keys, setKeys] = useState({
    anthropic: '',
    modal: ''
  });
  
  useEffect(() => {
    // Boot WebContainer on load
    testWebContainer().then(() => {
      setStatus('WebContainer ready!');
    });
  }, []);
  
  const handleSaveKeys = async () => {
    await fetch('http://localhost:3000/settings/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(keys)
    });
    setStatus('Keys saved!');
  };
  
  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>TabRL - Setup Test</h1>
      <p>Status: {status}</p>
      
      <h2>1. Configure API Keys</h2>
      <input 
        placeholder="Anthropic API Key"
        value={keys.anthropic}
        onChange={(e) => setKeys({...keys, anthropic: e.target.value})}
        style={{ width: '300px', marginRight: '10px' }}
      />
      <input 
        placeholder="Modal Token"  
        value={keys.modal}
        onChange={(e) => setKeys({...keys, modal: e.target.value})}
        style={{ width: '300px', marginRight: '10px' }}
      />
      <button onClick={handleSaveKeys}>Save Keys</button>
      
      <h2>2. Test Modal Connection</h2>
      <button onClick={async () => {
        const resp = await fetch('http://localhost:3000/test/modal', {
          method: 'POST'
        });
        const data = await resp.json();
        setStatus(`Modal test: ${JSON.stringify(data)}`);
      }}>Test Modal</button>
      
      <h2>Next Steps</h2>
      <ul>
        <li>✓ WebContainer boots</li>
        <li>✓ Settings save/load works</li>
        <li>◯ Modal connection test</li>
        <li>◯ Load MuJoCo scene</li>
        <li>◯ Run ONNX inference</li>
      </ul>
    </div>
  );
}

export default App;
