// Test WebContainer boot + settings save

import { WebContainer } from '@webcontainer/api';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';

// Singleton WebContainer instance
let webcontainerInstance = null;
let webcontainerURL = null;
let terminal = null;
let shellProcess = null;

// Export function to get current WebContainer URL
export function getWebContainerURL() {
  console.log('🔍 getWebContainerURL called, current value:', webcontainerURL);
  return webcontainerURL;
}

// Debug function
export function debugWebContainer() {
  console.log('🐛 WebContainer Debug Info:');
  console.log('- Instance:', !!webcontainerInstance);
  console.log('- URL:', webcontainerURL);
  console.log('- Instance type:', typeof webcontainerInstance);
  return {
    hasInstance: !!webcontainerInstance,
    url: webcontainerURL,
    instanceType: typeof webcontainerInstance
  };
}

// Force restart WebContainer
export function forceRestartWebContainer() {
  console.log('🔄 Forcing WebContainer restart...');
  webcontainerInstance = null;
  webcontainerURL = null;
  return testWebContainer();
}

export async function testWebContainer(apiKeys = {}) {
  try {
    // Reuse existing instance if available
    if (webcontainerInstance) {
      console.log('♻️ Using existing WebContainer instance');
      return webcontainerInstance;
    }
    
    console.log('🚀 Starting WebContainer boot...');
    webcontainerInstance = await WebContainer.boot();
    console.log('✅ WebContainer booted successfully');
    
    // Create simpler files structure to avoid DataCloneError
    const serverCode = `
const express = require('express');
const cors = require('cors');

const app = express();
const startTime = new Date().toISOString();

// Store API keys in memory
let API_KEYS = {};

// Enable CORS for all origins
app.use(cors({
  origin: '*',
  credentials: false
}));

// Add explicit CORS headers for COEP compatibility  
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.header('Cross-Origin-Resource-Policy', 'cross-origin');
  res.header('Cross-Origin-Embedder-Policy', 'credentialless');
  next();
});

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  console.log('Health check requested');
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// CORS test endpoint
app.get('/cors-test', (req, res) => {
  console.log('CORS test requested');
  res.json({ message: 'CORS is working!', timestamp: new Date().toISOString() });
});

// Start server
const PORT = 3000;
app.listen(PORT, '0.0.0.0', () => {
  console.log('🚀 Server running on http://0.0.0.0:' + PORT);
  console.log('✅ TabRL WebContainer ready for API calls');
});
`;

    const packageJson = `{
  "name": "tabrl-webcontainer",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  }
}`;

    const files = {
      'package.json': {
        file: {
          contents: packageJson
        }
      },
      'server.js': {
        file: {
          contents: serverCode
        }
      }
    };
    
    console.log('📁 Mounting files...');
    await webcontainerInstance.mount(files);
    console.log('✅ Files mounted');
    
    // Install npm packages
    console.log('📦 Installing npm packages...');
    const installProcess = await webcontainerInstance.spawn('npm', ['install']);
    
    // Stream install output
    installProcess.output.pipeTo(new WritableStream({
      write(data) {
        console.log('📦 npm install:', data);
      }
    }));
    
    const installExitCode = await installProcess.exit;
    console.log('📦 npm install completed with exit code: ' + installExitCode);
    
    if (installExitCode !== 0) {
      throw new Error('npm install failed with exit code ' + installExitCode);
    }
    
    // Start the server
    console.log('🚀 Starting Express server...');
    const serverProcess = await webcontainerInstance.spawn('npm', ['start']);
    
    // Stream server output
    serverProcess.output.pipeTo(new WritableStream({
      write(data) {
        console.log('🌐 Server:', data);
      }
    }));
    
    // Listen for server-ready event
    webcontainerInstance.on('server-ready', (port, url) => {
      console.log('✅ Server ready at ' + url + ' (port ' + port + ')');
      webcontainerURL = url;
      console.log('🔗 WebContainer URL stored:', webcontainerURL);
    });
    
    // Alternative: try to get URL directly if available
    setTimeout(() => {
      if (!webcontainerURL) {
        // Try to construct URL manually if server-ready event missed
        const baseUrl = 'https://k03e2io1v3fx9wvj0vr8qd5q58o56n-fkdo--3000--6ba59070.local-corp.webcontainer-api.io';
        webcontainerURL = baseUrl;
        console.log('🔗 Using manual URL:', webcontainerURL);
      }
    }, 2000);
    
    // Auto-configure API keys if provided
    if (apiKeys && Object.keys(apiKeys).length > 0) {
      console.log('🔑 Auto-configuring API keys...');
      // Wait longer for server to start with CORS configured
      setTimeout(async () => {
        try {
          console.log('🔑 Sending keys to WebContainer...');
          const response = await fetch(webcontainerURL + '/settings/save', {
            method: 'GET',
            params: apiKeys
          });
          const result = await response.json();
          console.log('✅ Keys configured:', result.keys_stored);
        } catch (e) {
          console.log('⏳ WebContainer server not ready yet:', e.message);
        }
      }, 6000); // Wait longer for CORS setup
    }
    
    return webcontainerInstance;
    
  } catch (error) {
    console.error('❌ WebContainer boot failed:', error);
    throw error;
  }
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
    method: 'GET',
    params: settings
  });
  
  console.log('Save response:', await saveResp.json());
  
  // Load
  const loadResp = await fetch('http://localhost:3000/settings/load');
  console.log('Loaded settings:', await loadResp.json());
}

// Terminal setup and shell functions
export function setupTerminal(terminalElement) {
  console.log('🖥️ setupTerminal called with element:', terminalElement);
  
  if (terminal) {
    console.log('♻️ Terminal already exists');
    return terminal;
  }

  if (!terminalElement) {
    console.error('❌ No terminal element provided');
    return null;
  }

  console.log('🖥️ Creating new terminal...');
  const fitAddon = new FitAddon();
  
  terminal = new Terminal({
    convertEol: true,
    cursorBlink: true,
    theme: {
      background: '#1e1e1e',
      foreground: '#ffffff'
    }
  });
  
  console.log('🖥️ Loading fit addon...');
  terminal.loadAddon(fitAddon);
  
  console.log('🖥️ Opening terminal in element...');
  terminal.open(terminalElement);
  
  console.log('🖥️ Fitting terminal...');
  fitAddon.fit();
  
  // Handle window resize
  const handleResize = () => {
    fitAddon.fit();
    if (shellProcess) {
      shellProcess.resize({
        cols: terminal.cols,
        rows: terminal.rows,
      });
    }
  };
  
  window.addEventListener('resize', handleResize);
  
  console.log('✅ Terminal setup complete');
  return terminal;
}

export async function startShell() {
  console.log('🐚 startShell called');
  
  if (!webcontainerInstance) {
    console.error('❌ WebContainer not available');
    return null;
  }
  
  if (!terminal) {
    console.error('❌ Terminal not setup');
    return null;
  }
  
  if (shellProcess) {
    console.log('♻️ Shell already running');
    return shellProcess;
  }
  
  try {
    console.log('🐚 Starting WebContainer shell...');
    
    shellProcess = await webcontainerInstance.spawn('jsh', {
      terminal: {
        cols: terminal.cols,
        rows: terminal.rows,
      },
    });
    
    console.log('🐚 Shell process created, connecting streams...');
    
    // Connect shell output to terminal
    shellProcess.output.pipeTo(
      new WritableStream({
        write(data) {
          console.log('📤 Shell output:', data);
          terminal.write(data);
        },
      })
    );
    
    // Connect terminal input to shell
    const input = shellProcess.input.getWriter();
    terminal.onData((data) => {
      console.log('📥 Terminal input:', data);
      input.write(data);
    });
    
    console.log('✅ WebContainer shell started and connected');
    return shellProcess;
    
  } catch (error) {
    console.error('❌ Failed to start shell:', error);
    return null;
  }
}

export function getTerminal() {
  return terminal;
}

export function getShellProcess() {
  return shellProcess;
}

// ===== WebContainer-Native Functions =====
// Direct file system and process management instead of HTTP

export async function saveApiKeysToContainer(apiKeys) {
  console.log('💾 Saving API keys directly to WebContainer...');
  
  if (!webcontainerInstance) {
    throw new Error('WebContainer not available');
  }

  // Create .env file content
  const envContent = `# TabRL API Keys
ANTHROPIC_API_KEY=${apiKeys.anthropic_api_key || ''}
MODAL_TOKEN_ID=${apiKeys.modal_token_id || ''}
MODAL_TOKEN_SECRET=${apiKeys.modal_token_secret || ''}
`;

  try {
    // Write directly to file system
    await webcontainerInstance.fs.writeFile('/.env', envContent);
    console.log('✅ API keys saved to /.env');
    return { success: true, message: 'API keys saved successfully' };
  } catch (error) {
    console.error('❌ Failed to save API keys:', error);
    throw error;
  }
}

export async function loadApiKeysFromContainer() {
  console.log('📖 Loading API keys from WebContainer...');
  
  if (!webcontainerInstance) {
    throw new Error('WebContainer not available');
  }

  try {
    // Read directly from file system
    const envContent = await webcontainerInstance.fs.readFile('/.env', 'utf8');
    
    // Parse .env content
    const keys = {};
    envContent.split('\n').forEach(line => {
      if (line.includes('=') && !line.startsWith('#')) {
        const [key, value] = line.split('=', 2);
        keys[key.toLowerCase()] = value;
      }
    });
    
    console.log('✅ API keys loaded from /.env');
    return {
      anthropic_api_key: keys.anthropic_api_key || '',
      modal_token_id: keys.modal_token_id || '',
      modal_token_secret: keys.modal_token_secret || ''
    };
  } catch (error) {
    if (error.code === 'ENOENT') {
      console.log('📝 No .env file found, returning empty keys');
      return { anthropic_api_key: '', modal_token_id: '', modal_token_secret: '' };
    }
    console.error('❌ Failed to load API keys:', error);
    throw error;
  }
}

export async function testModalApi() {
  console.log('🧪 Testing Modal API via Python script...');
  
  if (!webcontainerInstance) {
    throw new Error('WebContainer not available');
  }

  // First, load API keys from .env file
  let envVars = {};
  try {
    const envContent = await webcontainerInstance.fs.readFile('/.env', 'utf8');
    envContent.split('\n').forEach(line => {
      if (line.includes('=') && !line.startsWith('#')) {
        const [key, value] = line.split('=', 2);
        envVars[key.trim()] = value.trim();
      }
    });
    console.log('📖 Loaded environment variables from .env');
  } catch (error) {
    console.log('⚠️ No .env file found, using empty environment');
  }

  // Create test script
  const testScript = `#!/usr/bin/env python3
import os
import sys

print("🧪 Testing Modal API connection...")

# Load API keys from environment
modal_token_id = os.getenv('MODAL_TOKEN_ID')
modal_token_secret = os.getenv('MODAL_TOKEN_SECRET')

if not modal_token_id or not modal_token_secret:
    print("❌ Modal API keys not found in environment")
    print("Available env vars:", list(os.environ.keys()))
    sys.exit(1)

print(f"✅ Modal Token ID found: {modal_token_id[:8]}...")
print(f"✅ Modal Token Secret found: {modal_token_secret[:8]}...")

# Test API connection (replace with actual Modal API endpoint)
try:
    print("✅ Modal API keys loaded successfully")
    print("📡 Ready for Modal deployment")
    print("🎯 Next: Implement actual Modal API calls")
except Exception as e:
    print(f"❌ Modal API test failed: {e}")
    sys.exit(1)
`;

  try {
    // Write test script
    await webcontainerInstance.fs.writeFile('/test_modal.py', testScript);
    console.log('📝 Created test_modal.py');

    // Execute the script with environment variables
    const spawnedProcess = await webcontainerInstance.spawn('python3', ['/test_modal.py'], {
      env: envVars  // Pass the loaded environment variables
    });

    // Collect output
    let output = '';
    const outputStream = new WritableStream({
      write(data) {
        output += data;
        console.log('🐍 Python:', data);
      }
    });

    await spawnedProcess.output.pipeTo(outputStream);
    const exitCode = await spawnedProcess.exit;

    return {
      success: exitCode === 0,
      output: output,
      exitCode: exitCode
    };

  } catch (error) {
    console.error('❌ Modal API test failed:', error);
    throw error;
  }
}

export async function executeScript(scriptPath, args = [], options = {}) {
  console.log(`🐍 Executing script: ${scriptPath}`, args);
  
  if (!webcontainerInstance) {
    throw new Error('WebContainer not available');
  }

  try {
    const process = await webcontainerInstance.spawn('python3', [scriptPath, ...args], {
      env: options.env || {}
    });

    // Stream output in real-time
    let output = '';
    const outputStream = new WritableStream({
      write(data) {
        output += data;
        console.log('🐍 Script output:', data);
        // Could emit events here for real-time UI updates
      }
    });

    await process.output.pipeTo(outputStream);
    const exitCode = await process.exit;

    return {
      success: exitCode === 0,
      output: output,
      exitCode: exitCode
    };

  } catch (error) {
    console.error('❌ Script execution failed:', error);
    throw error;
  }
}
