#!/usr/bin/env python3
"""
Setup script for CrewAI SQL Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "config",
        "data", 
        "src",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up CrewAI SQL Agent...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check if virtual environment is active
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Warning: Virtual environment not detected")
        print("   Consider creating and activating a virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install and start Ollama: https://ollama.ai/")
    print("2. Pull the tinyllama model: ollama pull tinyllama:1.1b-chat")
    print("3. Set up PostgreSQL database")
    print("4. Update config/database_config.json with your database credentials")
    print("5. Run the application: streamlit run app.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 