import React, { useState } from 'react';
import './App.css';
import MujocoTest from './components/MujocoTest';
import MujocoViewer from './components/MujocoViewer';
import TrainingPageV2 from './components/TrainingPageV2';
import MultiViewerDemo from './components/MultiViewerDemo';

function App() {
  const [activeTab, setActiveTab] = useState('viewer');

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 TabRL: Tabular Reinforcement Learning</h1>
        <p className="subtitle">Train robots using natural language instructions</p>
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
        <button 
          className={activeTab === 'multiviewer' ? 'active' : ''}
          onClick={() => setActiveTab('multiviewer')}
        >
          Multi Viewer Demo
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'viewer' && (
          <ModelViewer />
        )}
        {activeTab === 'inference' && (
          <InferencePanel />
        )}
        {activeTab === 'training' && (
          <TrainingPageV2 />
        )}
        {activeTab === 'multiviewer' && (
          <MultiViewerDemo />
        )}
      </main>
    </div>
  );
}

function ModelViewer() {
  return (
    <div className="panel">
      <h2>🎬 Model Viewer</h2>
      <MujocoTest />
      <div className="model-viewport">
        <MujocoViewer />
      </div>
      <div className="model-controls">
        <button>Load Robot XML</button>
        <button>Reset View</button>
        <button>Play Animation</button>
      </div>
    </div>
  );
}

function InferencePanel() {
  return (
    <div className="panel">
      <h2>🧠 Inference</h2>
      <p>Run inference with loaded models</p>
      <div className="inference-form">
        <label>
          Model:
          <select>
            <option>Select a trained model...</option>
          </select>
        </label>
        <label>
          Input:
          <textarea placeholder="Enter observation data..." rows={4} />
        </label>
        <button className="primary-button">Run Inference</button>
      </div>
      <div className="inference-results">
        <h3>Results</h3>
        <pre>No results yet...</pre>
      </div>
    </div>
  );
}

export default App;
