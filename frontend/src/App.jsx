import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { 
  testWebContainer, 
  getWebContainerURL, 
  debugWebContainer,
  setupTerminal,
  startShell,
  saveApiKeysToContainer,
  loadApiKeysFromContainer,
  testModalApi
} from './webcontainer-test.js';

function App() {
  const [status, setStatus] = useState('Starting...');
  const [webContainerReady, setWebContainerReady] = useState(false);
  const [apiKeys, setApiKeys] = useState({
    anthropic_api_key: '',
    modal_token_id: '',
    modal_token_secret: ''
  });
  const [showTerminal, setShowTerminal] = useState(false);
  const terminalRef = useRef(null);
  
  useEffect(() => {
    // Boot WebContainer without keys initially
    testWebContainer().then(() => {
      setStatus('WebContainer ready! Paste your API keys below.');
      setWebContainerReady(true);
    }).catch((err) => {
      setStatus(`WebContainer failed: ${err.message}`);
    });
  }, []);

  const getURL = () => {
    const url = getWebContainerURL();
    if (!url) {
      throw new Error('WebContainer URL not available yet. Please wait a moment.');
    }
    return url;
  };

  const handleSaveKeys = async () => {
    if (!webContainerReady) {
      alert('WebContainer not ready yet!');
      return;
    }

    try {
      await saveApiKeysToContainer(apiKeys);
      alert('Keys saved!');
    } catch (error) {
      alert(`Failed to save keys: ${error.message}`);
    }
  };

  const handleTestModal = async () => {
    if (!webContainerReady) {
      alert('WebContainer not ready yet!');
      return;
    }

    try {
      const result = await testModalApi();
      alert(`Modal test result: ${JSON.stringify(result, null, 2)}`);
    } catch (error) {
      alert(`Modal test failed: ${error.message}`);
    }
  };

  const handleLoadSettings = async () => {
    if (!webContainerReady) {
      alert('WebContainer not ready yet!');
      return;
    }

    try {
      const result = await loadApiKeysFromContainer();
      alert(`Settings: ${JSON.stringify(result, null, 2)}`);
    } catch (error) {
      alert(`Failed to load settings: ${error.message}`);
    }
  };

  const handleCORSTest = async () => {
    if (!webContainerReady) {
      alert('WebContainer not ready yet!');
      return;
    }

    try {
      const webContainerURL = getURL();
      console.log('Testing CORS with URL:', webContainerURL);
      const response = await fetch(`${webContainerURL}/cors-test`);
      const result = await response.json();
      alert(`CORS Test Success: ${JSON.stringify(result, null, 2)}`);
    } catch (error) {
      alert(`CORS Test Failed: ${error.message}`);
      console.error('CORS Error:', error);
    }
  };

  const handleDebug = async () => {
    if (!webContainerReady) {
      alert('WebContainer not ready yet!');
      return;
    }

    try {
      await debugWebContainer();
      alert('Debugging WebContainer...');
    } catch (error) {
      alert(`Failed to debug WebContainer: ${error.message}`);
    }
  };

  const handleToggleTerminal = () => {
    console.log('🖥️ Toggle terminal clicked, current state:', showTerminal);
    setShowTerminal(!showTerminal);
  };

  // Setup terminal after it's rendered
  useEffect(() => {
    if (showTerminal && terminalRef.current) {
      console.log('🖥️ Terminal div rendered, setting up terminal...');
      console.log('🖥️ Terminal ref element:', terminalRef.current);
      
      // Small delay to ensure DOM is fully ready
      setTimeout(async () => {
        try {
          console.log('🖥️ Calling setupTerminal...');
          const terminalInstance = setupTerminal(terminalRef.current);
          
          if (!terminalInstance) {
            console.error('❌ setupTerminal returned null');
            return;
          }
          
          console.log('🖥️ Terminal setup successful, starting shell...');
          const shell = await startShell();
          
          if (!shell) {
            console.error('❌ startShell returned null');
            return;
          }
          
          console.log('✅ Terminal ready for use');
        } catch (error) {
          console.error('❌ Failed to setup terminal:', error);
        }
      }, 100);
    }
  }, [showTerminal]); // Run when showTerminal changes

  return (
    <div className="App">
      <h1>TabRL WebContainer Test</h1>
      <p>Status: {status}</p>
      
      <div style={{ margin: '20px 0' }}>
        <h3>API Keys Configuration</h3>
        
        <div style={{ marginBottom: '10px' }}>
          <label>
            Anthropic API Key:
            <input
              type="password"
              value={apiKeys.anthropic_api_key}
              onChange={(e) => setApiKeys({...apiKeys, anthropic_api_key: e.target.value})}
              placeholder="sk-ant-..."
              style={{ marginLeft: '10px', width: '300px' }}
            />
          </label>
        </div>

        <div style={{ marginBottom: '10px' }}>
          <label>
            Modal Token ID:
            <input
              type="password"
              value={apiKeys.modal_token_id}
              onChange={(e) => setApiKeys({...apiKeys, modal_token_id: e.target.value})}
              placeholder="ak-..."
              style={{ marginLeft: '10px', width: '300px' }}
            />
          </label>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>
            Modal Token Secret:
            <input
              type="password"
              value={apiKeys.modal_token_secret}
              onChange={(e) => setApiKeys({...apiKeys, modal_token_secret: e.target.value})}
              placeholder="as-..."
              style={{ marginLeft: '10px', width: '300px' }}
            />
          </label>
        </div>

        <button 
          onClick={handleSaveKeys} 
          disabled={!webContainerReady}
          style={{ marginRight: '10px' }}
        >
          Save Keys to WebContainer
        </button>
        
        <button 
          onClick={handleTestModal} 
          disabled={!webContainerReady}
          style={{ marginRight: '10px' }}
        >
          Test Modal Connection
        </button>
        
        <button 
          onClick={handleLoadSettings} 
          disabled={!webContainerReady}
          style={{ marginRight: '10px' }}
        >
          Load Settings
        </button>
        
        <button 
          onClick={handleCORSTest} 
          disabled={!webContainerReady}
          style={{ marginRight: '10px' }}
        >
          Test CORS
        </button>
        
        <button 
          onClick={handleDebug} 
          disabled={!webContainerReady}
          style={{ marginRight: '10px' }}
        >
          Debug WebContainer
        </button>
        
        <button 
          onClick={handleToggleTerminal} 
          disabled={!webContainerReady}
        >
          Toggle Terminal
        </button>
      </div>
      {showTerminal && (
        <div ref={terminalRef} style={{ 
          width: '100%', 
          height: '300px', 
          border: '1px solid black',
          textAlign: 'left'
        }} />
      )}
    </div>
  )
}

export default App
