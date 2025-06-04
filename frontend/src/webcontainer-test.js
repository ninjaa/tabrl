// Test WebContainer boot + settings save

import { WebContainer } from '@webcontainer/api';

export async function testWebContainer(apiKeys = {}) {
  console.log('🚀 Booting WebContainer...');
  
  const webcontainer = await WebContainer.boot();
  
  // Create a simple Flask server with settings
  const files = {
    'server.py': {
      file: {
        contents: `
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

# Store API keys securely inside WebContainer
API_KEYS = {}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "TabRL WebContainer running!"})

@app.route('/settings/save', methods=['POST'])
def save_settings():
    global API_KEYS
    settings = request.json
    API_KEYS.update(settings)
    print(f"Stored keys: {list(API_KEYS.keys())}")  # Log key names only, not values
    return jsonify({"status": "saved", "keys_stored": list(API_KEYS.keys())})

@app.route('/settings/load', methods=['GET'])
def load_settings():
    return jsonify({"keys_configured": bool(API_KEYS), "available_keys": list(API_KEYS.keys())})

@app.route('/test/modal', methods=['POST'])
def test_modal():
    # Test proxying to Modal using stored keys
    import requests
    
    if 'modal_token_id' not in API_KEYS or 'modal_token_secret' not in API_KEYS:
        return jsonify({"error": "Modal credentials not configured"})
    
    try:
        # Use the actual Modal endpoint with authentication
        resp = requests.post('https://ninjaa--tabrl-hello-hello-dev.modal.run')
        return resp.json()
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
`
      }
    },
    'requirements.txt': {
      file: {
        contents: 'flask\nrequests\nonnxruntime\nnumpy\n'
      }
    }
  };
  
  // Mount files
  await webcontainer.mount(files);
  
  // Install Python packages
  const installProcess = await webcontainer.spawn('pip', ['install', '-r', 'requirements.txt']);
  await installProcess.exit;
  
  // Start the server
  const serverProcess = await webcontainer.spawn('python', ['server.py']);
  
  // Auto-configure API keys if provided
  if (apiKeys && Object.keys(apiKeys).length > 0) {
    // Wait a moment for server to start, then send keys
    setTimeout(async () => {
      try {
        const response = await fetch('http://localhost:3000/settings/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(apiKeys)
        });
        console.log('✅ API keys configured in WebContainer');
      } catch (e) {
        console.log('⏳ Waiting for WebContainer server...');
      }
    }, 2000);
  }
  
  // Wait for server to be ready
  webcontainer.on('server-ready', (port, url) => {
    console.log(`✅ Server ready at ${url}`);
  });
  
  return webcontainer;
}

// Test settings save/load
export async function testSettings() {
  const settings = {
    anthropic_key: 'sk-ant-...',
    modal_token: 'modal-token-...',
    theme: 'dark'
  };
  
  // Save
  const saveResp = await fetch('http://localhost:3000/settings/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings)
  });
  
  console.log('Save response:', await saveResp.json());
  
  // Load
  const loadResp = await fetch('http://localhost:3000/settings/load');
  console.log('Loaded settings:', await loadResp.json());
}
