import React, { useState, useEffect } from 'react';
import './TrainingPageV2.css';

const TrainingPageV2 = () => {
  // Model Selection State
  const [selectedModel, setSelectedModel] = useState(null);
  const [models, setModels] = useState([]);
  
  // Training Configuration
  const [behaviorPrompt, setBehaviorPrompt] = useState('');
  const [numSteps, setNumSteps] = useState(50000);
  
  // LLM Tabs State - Updated for 4 models
  const [activeLLMTab, setActiveLLMTab] = useState('claude');
  const [llmApproaches, setLLMApproaches] = useState({
    claude: null,
    openai: null,
    gemini: null,
    deepseek: null
  });
  
  // Training State
  const [trainingJobs, setTrainingJobs] = useState({
    claude: [],
    openai: [],
    gemini: [],
    deepseek: []
  });
  
  // Video Results
  const [videos, setVideos] = useState({});
  
  // Loading States
  const [loadingModels, setLoadingModels] = useState(true);
  const [generatingApproaches, setGeneratingApproaches] = useState({
    claude: false,
    openai: false,
    gemini: false,
    deepseek: false
  });

  // Fetch available models on mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch('http://localhost:8000/api/playground/environments');
      const data = await response.json();
      
      // Helper to format environment names nicely
      const formatName = (name) => {
        // Add spaces before capitals and format special cases
        return name
          .replace(/([A-Z])/g, ' $1')
          .replace(/Joystick/g, '')
          .replace(/FlatTerrain/g, '(Flat)')
          .replace(/RoughTerrain/g, '(Rough)')
          .replace(/  +/g, ' ')
          .trim();
      };
      
      // Map robot types to emojis
      const getRobotEmoji = (name, category) => {
        if (category === 'manipulation') {
          if (name.includes('Hand') || name.includes('Leap')) return '🤏';
          if (name.includes('Panda')) return '🦾';
          if (name.includes('Aloha')) return '🤖';
          return '🦾';
        }
        // Locomotion robots
        if (name.includes('Go1') || name.includes('Spot') || name.includes('Barkour')) return '🐕';
        if (name.includes('Humanoid') || name.includes('H1') || name.includes('G1')) return '🚶';
        if (name.includes('Op3')) return '🤖';
        if (name.includes('T1')) return '🦿';
        return '🤖';
      };
      
      // Transform the environments into a flat list of models
      const modelsList = [];
      
      // Add locomotion models
      if (data.locomotion) {
        data.locomotion.forEach(env => {
          modelsList.push({
            id: `locomotion/${env}`,
            name: formatName(env),
            rawName: env,
            category: 'locomotion',
            emoji: getRobotEmoji(env, 'locomotion'),
            thumbnail: '/images/robot_placeholder.jpg',
            description: `Locomotion robot: ${formatName(env)}`,
            pretrained: env.includes('Joystick') // Mark joystick controllers as pretrained
          });
        });
      }
      
      // Add manipulation models
      if (data.manipulation) {
        data.manipulation.forEach(env => {
          modelsList.push({
            id: `manipulation/${env}`,
            name: formatName(env),
            rawName: env,
            category: 'manipulation',
            emoji: getRobotEmoji(env, 'manipulation'),
            thumbnail: '/images/robot_placeholder.jpg',
            description: `Manipulation task: ${formatName(env)}`,
            pretrained: false
          });
        });
      }
      
      setModels(modelsList);
      
      // Select Go1 flat terrain by default
      const defaultModel = modelsList.find(m => m.rawName === 'Go1JoystickFlatTerrain');
      if (defaultModel) setSelectedModel(defaultModel);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoadingModels(false);
    }
  };

  const generateApproachesForLLM = async (llmProvider) => {
    if (!selectedModel || !behaviorPrompt) return;
    
    setGeneratingApproaches(prev => ({ ...prev, [llmProvider]: true }));
    
    // Map LLM provider to model name
    const model_mapping = {
        "claude": "claude-opus-4-20250514",
        "openai": "o3",
        "gemini": "gemini/gemini-2.5-pro-preview-06-05",
        "deepseek": "deepseek/deepseek-r1-0528"
    };
    
    try {
      const response = await fetch('http://localhost:8000/api/training/approaches', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel.id,
          task_description: behaviorPrompt,
          num_approaches: 3,
          llm_model: model_mapping[llmProvider],
          num_steps: numSteps
        })
      });
      
      const data = await response.json();
      setLLMApproaches(prev => ({
        ...prev,
        [llmProvider]: data.approaches
      }));
    } catch (error) {
      console.error(`Failed to generate approaches for ${llmProvider}:`, error);
    } finally {
      setGeneratingApproaches(prev => ({ ...prev, [llmProvider]: false }));
    }
  };

  const generateAllApproaches = () => {
    // Generate approaches for all 4 LLM providers in parallel
    ['claude', 'openai', 'gemini', 'deepseek'].forEach(provider => {
      generateApproachesForLLM(provider);
    });
  };

  const startTrainingBatch = async (llmProvider) => {
    const approaches = llmApproaches[llmProvider];
    if (!approaches || approaches.length === 0) return;
    
    try {
      const response = await fetch('http://localhost:8000/api/training/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel.id,
          approaches: approaches,
          num_steps: numSteps,
          use_base_policy: selectedModel.has_base_policy
        })
      });
      
      const data = await response.json();
      setTrainingJobs(prev => ({
        ...prev,
        [llmProvider]: data.jobs
      }));
      
      // Start polling for each job
      data.jobs.forEach(job => {
        pollJobStatus(job.job_id, llmProvider);
      });
    } catch (error) {
      console.error(`Failed to start training for ${llmProvider}:`, error);
    }
  };

  const pollJobStatus = async (jobId, llmProvider) => {
    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/training/${jobId}/status`);
        const status = await response.json();
        
        // Update job status
        setTrainingJobs(prev => ({
          ...prev,
          [llmProvider]: prev[llmProvider].map(job => 
            job.job_id === jobId ? { ...job, ...status } : job
          )
        }));
        
        if (status.status === 'completed') {
          // Generate video
          await generateVideo(jobId, llmProvider);
        } else if (status.status === 'running') {
          // Continue polling
          setTimeout(poll, 2000);
        }
      } catch (error) {
        console.error(`Failed to poll status for ${jobId}:`, error);
      }
    };
    poll();
  };

  const generateVideo = async (jobId, llmProvider) => {
    try {
      const response = await fetch('http://localhost:8000/api/training/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          model_path: `/workspace/models/${jobId}_model.pkl`
        })
      });
      
      const data = await response.json();
      setVideos(prev => ({
        ...prev,
        [`${llmProvider}_${jobId}`]: data.video_url
      }));
    } catch (error) {
      console.error(`Failed to generate video for ${jobId}:`, error);
    }
  };

  const getTrainingTimeEstimate = () => {
    if (selectedModel?.has_base_policy) {
      if (numSteps <= 10000) return "~30 seconds";
      if (numSteps <= 50000) return "~2 minutes";
      return "~4 minutes";
    } else {
      if (numSteps <= 100000) return "~4 minutes";
      if (numSteps <= 250000) return "~10 minutes";
      return "~20 minutes";
    }
  };

  return (
    <div className="training-page-v2">
      {/* Model Gallery Section */}
      <section className="model-gallery">
        <h2>Select a Robot Model</h2>
        {loadingModels ? (
          <div className="loading">Loading models...</div>
        ) : (
          <div className="model-grid">
            {models.map(model => (
              <div 
                key={model.id} 
                className={`model-card ${selectedModel?.id === model.id ? 'selected' : ''}`}
                onClick={() => setSelectedModel(model)}
              >
                <div className="model-thumbnail">
                  <span className="robot-emoji">{model.emoji}</span>
                </div>
                <h3>{model.name}</h3>
                <p className="model-info">
                  <span className="category">{model.category}</span>
                  {model.pretrained && <span className="badge">Pre-trained</span>}
                </p>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Behavior Configuration */}
      <section className="behavior-config">
        <h2>Describe the Behavior</h2>
        <div className="config-form">
          <textarea
            value={behaviorPrompt}
            onChange={(e) => setBehaviorPrompt(e.target.value)}
            placeholder="Keep shaking your hips aloha dude! 🌺"
            rows={3}
          />
          
          <div className="training-steps">
            <label>Training Steps: {numSteps.toLocaleString()}</label>
            <input
              type="range"
              min={10000}
              max={100000}
              step={10000}
              value={numSteps}
              onChange={(e) => setNumSteps(Number(e.target.value))}
            />
            <div className="step-labels">
              <span>Quick</span>
              <span>Standard</span>
              <span>Quality</span>
            </div>
            <p className="time-estimate">
              Estimated time: {getTrainingTimeEstimate()} per approach
            </p>
          </div>
          
          <button 
            className="generate-all-btn"
            onClick={generateAllApproaches}
            disabled={!selectedModel || !behaviorPrompt}
          >
            Generate Training Approaches (All LLMs)
          </button>
        </div>
      </section>

      {/* Multi-LLM Training Section */}
      <section className="multi-llm-training">
        <h2>Training Approaches by LLM</h2>
        
        {/* LLM Provider Tabs - Updated with company branding */}
        <div className="llm-tabs">
          <button
            className={`tab ${activeLLMTab === 'claude' ? 'active' : ''}`}
            onClick={() => setActiveLLMTab('claude')}
          >
            <span className="company-icon">🟣</span>
            Claude Opus 4
          </button>
          <button
            className={`tab ${activeLLMTab === 'openai' ? 'active' : ''}`}
            onClick={() => setActiveLLMTab('openai')}
          >
            <span className="company-icon">🟢</span>
            OpenAI o3
          </button>
          <button
            className={`tab ${activeLLMTab === 'gemini' ? 'active' : ''}`}
            onClick={() => setActiveLLMTab('gemini')}
          >
            <span className="company-icon">🔵</span>
            Gemini Pro 2.5
          </button>
          <button
            className={`tab ${activeLLMTab === 'deepseek' ? 'active' : ''}`}
            onClick={() => setActiveLLMTab('deepseek')}
          >
            <span className="company-icon">🔴</span>
            DeepSeek R1
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {generatingApproaches[activeLLMTab] ? (
            <div className="loading">Generating approaches...</div>
          ) : llmApproaches[activeLLMTab] ? (
            <div className="approaches-section">
              <div className="approaches-grid">
                {llmApproaches[activeLLMTab].map((approach, idx) => (
                  <div key={idx} className="approach-card">
                    <h4>{approach.name}</h4>
                    <p>{approach.description}</p>
                    <details>
                      <summary>View Reward Code</summary>
                      <pre>{approach.reward_code}</pre>
                    </details>
                  </div>
                ))}
              </div>
              
              <button
                className="train-all-btn"
                onClick={() => startTrainingBatch(activeLLMTab)}
                disabled={trainingJobs[activeLLMTab].length > 0}
              >
                Train All 3 Approaches
              </button>
              
              {/* Training Progress */}
              {trainingJobs[activeLLMTab].length > 0 && (
                <div className="training-progress">
                  {trainingJobs[activeLLMTab].map((job, idx) => (
                    <div key={job.job_id} className="job-progress">
                      <h5>{llmApproaches[activeLLMTab][idx].name}</h5>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill"
                          style={{ width: `${(job.progress || 0) * 100}%` }}
                        />
                      </div>
                      <span className="status">{job.status}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              Generate approaches to see training options
            </div>
          )}
        </div>
      </section>

      {/* Video Comparison Section */}
      <section className="video-comparison">
        <h2>Results Comparison</h2>
        <div className="video-grid">
          {['claude', 'openai', 'gemini', 'deepseek'].map(provider => (
            <div key={provider} className="llm-videos">
              <h3>
                {provider === 'claude' && '🟣 Claude Opus 4'}
                {provider === 'openai' && '🟢 OpenAI o3'}
                {provider === 'gemini' && '🔵 Gemini Pro 2.5'}
                {provider === 'deepseek' && '🔴 DeepSeek R1'}
              </h3>
              <div className="videos-row">
                {trainingJobs[provider].map((job, idx) => {
                  const videoKey = `${provider}_${job.job_id}`;
                  const video = videos[videoKey];
                  
                  return (
                    <div key={job.job_id} className="video-container">
                      <h5>{llmApproaches[provider]?.[idx]?.name || 'Approach ' + (idx + 1)}</h5>
                      {video ? (
                        <video controls loop>
                          <source src={video} type="video/mp4" />
                        </video>
                      ) : job.status === 'completed' ? (
                        <div className="loading">Generating video...</div>
                      ) : (
                        <div className="placeholder">Training in progress...</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        
        {/* Playback Controls */}
        <div className="playback-controls">
          <button onClick={() => playAllVideos()}>▶ Play All</button>
          <button onClick={() => pauseAllVideos()}>⏸ Pause All</button>
          <button onClick={() => resetAllVideos()}>↻ Reset All</button>
        </div>
      </section>
    </div>
  );
};

const playAllVideos = () => {
  document.querySelectorAll('.video-comparison video').forEach(v => v.play());
};

const pauseAllVideos = () => {
  document.querySelectorAll('.video-comparison video').forEach(v => v.pause());
};

const resetAllVideos = () => {
  document.querySelectorAll('.video-comparison video').forEach(v => {
    v.currentTime = 0;
    v.play();
  });
};

export default TrainingPageV2;
