#!/usr/bin/env python3
"""
TabRL Backend Startup Script
Handles virtual environment setup and service launch
"""

import sys
import subprocess
import os
from pathlib import Path

def setup_venv():
    """Create and setup virtual environment"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Get pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install requirements
    print("📥 Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    
    return python_path

def start_server(python_path):
    """Start the FastAPI server"""
    print("🚀 Starting TabRL Backend Server...")
    print("📡 Frontend can connect to: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([
            str(python_path), "-m", "uvicorn", 
            "app:app", 
            "--host", "127.0.0.1", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 TabRL Backend stopped")

def main():
    """Main startup sequence"""
    os.chdir(Path(__file__).parent)
    
    print("🤖 TabRL Backend Setup")
    print("=" * 30)
    
    # Setup environment
    python_path = setup_venv()
    
    # Start server
    start_server(python_path)

if __name__ == "__main__":
    main()
