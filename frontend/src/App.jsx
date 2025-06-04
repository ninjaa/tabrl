import React, { useState } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('viewer');
  const [theme, setTheme] = useState('light'); // Default to light for demo

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <div className={`app ${theme}`}>
      <header className="app-header">
        <div className="header-content">
          <div className="title-section">
            <h1>🤖 TabRL</h1>
            <p>Robotics Training Platform</p>
          </div>
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      <nav className="app-nav">
        <button 
          className={activeTab === 'viewer' ? 'active' : ''} 
          onClick={() => setActiveTab('viewer')}
        >
          Model Viewer
        </button>
        <button 
          className={activeTab === 'inference' ? 'active' : ''} 
          onClick={() => setActiveTab('inference')}
        >
          Inference
        </button>
        <button 
          className={activeTab === 'training' ? 'active' : ''} 
          onClick={() => setActiveTab('training')}
        >
          Training
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'viewer' && <ModelViewer />}
        {activeTab === 'inference' && <InferencePanel />}
        {activeTab === 'training' && <TrainingPanel />}
      </main>
    </div>
  );
}

// Placeholder components - we'll implement these next
function ModelViewer() {
  return (
    <div className="panel">
      <h2>🎬 Model Viewer</h2>
      <div className="model-viewport">
        <p>MuJoCo WASM integration will go here</p>
        <div className="placeholder-3d">
          [3D Robot Visualization]
        </div>
      </div>
      <div className="model-controls">
        <button>Load Robot XML</button>
        <button>Reset Pose</button>
      </div>
    </div>
  );
}

function InferencePanel() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');

  const handleInference = async () => {
    setResponse('Connecting to local Python service...');
    // TODO: Connect to localhost:8000/inference
  };

  return (
    <div className="panel">
      <h2>🧠 Inference</h2>
      <div className="inference-chat">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe what you want the robot to do..."
          rows={3}
        />
        <button onClick={handleInference}>
          Generate Policy
        </button>
        <div className="response-area">
          {response && <pre>{response}</pre>}
        </div>
      </div>
    </div>
  );
}

function TrainingPanel() {
  return (
    <div className="panel">
      <h2>🏋️ Training</h2>
      <div className="training-controls">
        <p>Training pipeline controls will go here</p>
        <button>Start Training</button>
        <button>Stop Training</button>
        <div className="training-progress">
          <p>Ready to train</p>
        </div>
      </div>
    </div>
  );
}

export default App;
