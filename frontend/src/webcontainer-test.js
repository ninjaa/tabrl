// Test WebContainer boot + settings save

import { WebContainer } from '@webcontainer/api';

export async function testWebContainer() {
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

# Simple file-based settings
SETTINGS_FILE = '/tmp/settings.json'

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "TabRL WebContainer running!"})

@app.route('/settings/save', methods=['POST'])
def save_settings():
    settings = request.json
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)
    return jsonify({"status": "saved"})

@app.route('/settings/load', methods=['GET'])
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/test/modal', methods=['POST'])
def test_modal():
    # Test proxying to Modal
    import requests
    
    # This will work because WebContainer can make external requests!
    try:
        # Replace with your Modal endpoint
        resp = requests.post('https://YOUR_MODAL_APP.modal.run/hello')
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
