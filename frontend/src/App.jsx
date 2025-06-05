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
  const [taskDescription, setTaskDescription] = useState('');
  const [selectedScene, setSelectedScene] = useState('manipulation/universal_robots_ur5e');
  const [selectedModel, setSelectedModel] = useState('claude-3-5-sonnet-20241022');
  const [episodes, setEpisodes] = useState(50);
  
  // Policy generation state
  const [rewardApproaches, setRewardApproaches] = useState([]);
  const [selectedReward, setSelectedReward] = useState(null);
  const [generatingPolicies, setGeneratingPolicies] = useState(false);
  
  // Training state
  const [activeTrainings, setActiveTrainings] = useState([]);
  const [trainingLogs, setTrainingLogs] = useState({});

  const scenes = [
    'manipulation/universal_robots_ur5e',
    'locomotion/anymal_c',
    'humanoids/unitree_g1',
    'hands/shadow_hand'
  ];

  const models = [
    'claude-3-5-sonnet-20241022',
    'gpt-4o',
    'gemini-1.5-pro',
    'deepseek-chat'
  ];

  const generatePolicies = async () => {
    if (!taskDescription.trim()) {
      alert('Please enter a task description');
      return;
    }

    console.log('🚀 Starting policy generation...', {
      taskDescription,
      selectedScene,
      selectedModel
    });

    setGeneratingPolicies(true);
    try {
      const response = await fetch('http://localhost:8000/api/policy/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: taskDescription,
          scene_name: selectedScene,
          model: selectedModel
        })
      });

      console.log('📡 Response status:', response.status);
      const data = await response.json();
      console.log('📦 Response data:', data);

      if (response.ok) {
        console.log('✅ Policy generation successful:', data.reward_functions?.length || 0, 'reward functions');
        
        // Parse raw_response if reward_functions are embedded there
        let rewardFunctions = data.reward_functions || [];
        if (!rewardFunctions.length && data.raw_response) {
          try {
            const parsedResponse = JSON.parse(data.raw_response);
            rewardFunctions = parsedResponse.reward_functions || [];
            console.log('📋 Parsed from raw_response:', rewardFunctions.length, 'reward functions');
          } catch (e) {
            console.error('Failed to parse raw_response:', e);
          }
        }
        
        setRewardApproaches(rewardFunctions);
        setSelectedReward(null);
      } else {
        console.error('❌ Policy generation failed:', data);
        alert(`Policy generation failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('🔥 Network error:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setGeneratingPolicies(false);
    }
  };

  const startTraining = async () => {
    if (!selectedReward) {
      alert('Please select a reward approach first');
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_description: taskDescription,
          scene_name: selectedScene,
          episodes: episodes,
          reward_code: selectedReward.reward
        })
      });

      const data = await response.json();
      if (response.ok) {
        const newTraining = {
          id: data.training_id,
          task: taskDescription,
          scene: selectedScene,
          reward_name: selectedReward.name,
          status: 'started',
          progress: 0,
          episode: 0,
          reward: 0,
          loss: null
        };
        
        setActiveTrainings(prev => [...prev, newTraining]);
        pollTrainingStatus(data.training_id);
      } else {
        alert(`Training failed: ${data.detail}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  const pollTrainingStatus = async (trainingId) => {
    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/training/${trainingId}/status`);
        const status = await response.json();
        
        if (response.ok) {
          setActiveTrainings(prev => 
            prev.map(t => t.id === trainingId ? { ...t, ...status } : t)
          );

          // Continue polling if still training
          if (status.status === 'training' || status.status === 'initializing') {
            setTimeout(poll, 2000);
          }
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    };
    
    poll();
  };

  return (
    <div className="panel">
      <h2>🏋️ Training Pipeline</h2>
      
      {/* Step 1: Task & Scene Configuration */}
      <div className="training-step">
        <h3>1. Task Configuration</h3>
        <div className="form-group">
          <label>Task Description:</label>
          <textarea
            value={taskDescription}
            onChange={(e) => setTaskDescription(e.target.value)}
            placeholder="Describe what you want the robot to do..."
            rows={3}
          />
        </div>
        
        <div className="form-row">
          <div className="form-group">
            <label>Scene:</label>
            <select value={selectedScene} onChange={(e) => setSelectedScene(e.target.value)}>
              {scenes.map(scene => (
                <option key={scene} value={scene}>{scene}</option>
              ))}
            </select>
          </div>
          
          <div className="form-group">
            <label>LLM Model:</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {models.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
          
          <div className="form-group">
            <label>Episodes:</label>
            <input
              type="number"
              value={episodes}
              onChange={(e) => setEpisodes(Number(e.target.value))}
              min="1"
              max="1000"
            />
          </div>
        </div>
        
        <button 
          onClick={generatePolicies} 
          disabled={generatingPolicies || !taskDescription.trim()}
          className="primary-button"
        >
          {generatingPolicies ? 'Generating Reward Approaches...' : 'Generate Reward Approaches'}
        </button>
      </div>

      {/* Step 2: Reward Approach Selection */}
      {rewardApproaches.length > 0 && (
        <div className="training-step">
          <h3>2. Select Reward Approach</h3>
          <p>Found {rewardApproaches.length} reward functions:</p>
          <div className="reward-approaches">
            {rewardApproaches.map((reward, index) => (
              <div 
                key={index}
                className={`reward-card ${selectedReward === reward ? 'selected' : ''}`}
                onClick={() => setSelectedReward(reward)}
              >
                <h4>{reward.name || `Reward Function ${index + 1}`}</h4>
                <p className="reward-type">{reward.type || 'Unknown'}</p>
                <p className="reward-description">
                  {reward.description || `${reward.type} reward approach for ${reward.name}`}
                </p>
                <details>
                  <summary>View Code</summary>
                  <pre className="reward-code">{reward.reward || 'No code available'}</pre>
                </details>
              </div>
            ))}
          </div>
          
          <button 
            onClick={startTraining} 
            disabled={!selectedReward}
            className="primary-button"
          >
            Start Training with {selectedReward?.name}
          </button>
        </div>
      )}

      {/* Step 3: Training Progress */}
      {activeTrainings.length > 0 && (
        <div className="training-step">
          <h3>3. Training Progress</h3>
          <div className="training-jobs">
            {activeTrainings.map(training => (
              <div key={training.id} className="training-job">
                <div className="job-header">
                  <h4>{training.task}</h4>
                  <span className={`status ${training.status}`}>{training.status}</span>
                </div>
                
                <div className="job-details">
                  <p><strong>Scene:</strong> {training.scene}</p>
                  <p><strong>Reward:</strong> {training.reward_name}</p>
                  <p><strong>ID:</strong> {training.id}</p>
                </div>
                
                {training.status === 'training' && (
                  <div className="progress-section">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${(training.progress || 0) * 100}%` }}
                      ></div>
                    </div>
                    <div className="progress-stats">
                      <span>Episode: {training.episode}/{episodes}</span>
                      <span>Reward: {training.reward?.toFixed(2) || 0}</span>
                      <span>Loss: {training.loss?.toFixed(4) || 'N/A'}</span>
                      <span>ETA: {training.eta_seconds || 0}s</span>
                    </div>
                  </div>
                )}
                
                {training.status === 'completed' && (
                  <div className="completion-section">
                    <p>✅ Training completed!</p>
                    <p><strong>Final Reward:</strong> {training.reward?.toFixed(2)}</p>
                    {training.model_path && (
                      <p><strong>Model:</strong> {training.model_path}</p>
                    )}
                  </div>
                )}
                
                {training.status === 'failed' && (
                  <div className="error-section">
                    <p>❌ Training failed</p>
                    {training.error && <p><strong>Error:</strong> {training.error}</p>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
