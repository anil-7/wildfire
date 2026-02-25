#!/usr/bin/env python
"""
Quick Start Script
Sets up the environment and guides user through initial setup
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

import os
import shutil

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def setup_environment_file():
    """Setup .env file from example"""
    env_file = PROJECT_ROOT / "config" / ".env"
    env_example = PROJECT_ROOT / "config" / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("\n📝 Setting up environment file...")
        shutil.copy(env_example, env_file)
        print(f"✅ Created .env file at: {env_file}")
        print("\n⚠️  IMPORTANT: Please edit config/.env and add your:")
        print("   1. GROQ_API_KEY (for AI insights)")
        return False
    elif env_file.exists():
        print("✅ Environment file already exists")
        return True
    return False

def check_kaggle_credentials():
    """Check if Kaggle credentials exist"""
    kaggle_file = PROJECT_ROOT / "config" / "kaggle.json"
    
    if not kaggle_file.exists():
        print("\n⚠️  Kaggle credentials not found")
        print(f"   Please place your kaggle.json file at: {kaggle_file}")
        print("\n   How to get kaggle.json:")
        print("   1. Go to https://www.kaggle.com/")
        print("   2. Click on your profile picture → Account")
        print("   3. Scroll to 'API' section")
        print("   4. Click 'Create New Token'")
        print("   5. Save the downloaded kaggle.json to config/")
        return False
    
    print("✅ Kaggle credentials found")
    return True

def check_groq_api_key():
    """Check if Groq API key is configured"""
    env_file = PROJECT_ROOT / "config" / ".env"
    
    if not env_file.exists():
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    if 'your_groq_api_key_here' in content or 'GROQ_API_KEY=' not in content:
        print("\n⚠️  Groq API key not configured")
        print("   Please edit config/.env and add your GROQ_API_KEY")
        print("\n   How to get Groq API key:")
        print("   1. Go to https://console.groq.com/")
        print("   2. Sign up / Log in")
        print("   3. Go to API Keys section")
        print("   4. Create a new API key")
        print("   5. Copy the key and paste it in config/.env")
        return False
    
    print("✅ Groq API key configured")
    return True

def install_dependencies():
    """Prompt to install dependencies"""
    print("\n📦 Install Dependencies")
    print("   To install required packages, run:")
    print("   pip install -r requirements.txt")
    
    choice = input("\n   Install now? (y/n) [y]: ").strip().lower() or 'y'
    
    if choice == 'y':
        import subprocess
        print("\n   Installing dependencies...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Error installing dependencies")
            return False
    else:
        print("⚠️  Skipped. Install manually with: pip install -r requirements.txt")
        return False

def show_next_steps(has_kaggle, has_groq):
    """Show next steps based on configuration status"""
    print_header("🎯 NEXT STEPS")
    
    print("Step-by-step guide to get started:\n")
    
    print("1️⃣  CONFIGURE CREDENTIALS")
    if not has_kaggle:
        print("    ❌ Add kaggle.json to config/")
    else:
        print("    ✅ Kaggle credentials ready")
    
    if not has_groq:
        print("    ❌ Add GROQ_API_KEY to config/.env")
    else:
        print("    ✅ Groq API key ready")
    
    print("\n2️⃣  DOWNLOAD DATASETS")
    print("    python src/data_loader/download_datasets.py")
    
    print("\n3️⃣  PREPROCESS DATA")
    print("    python src/preprocessing/preprocess_data.py")
    
    print("\n4️⃣  TRAIN MODELS")
    print("    python src/training/train_detection_model.py")
    
    print("\n5️⃣  RUN APPLICATION")
    print("    Option A - Web Dashboard:")
    print("    python app/main.py --dashboard")
    print("\n    Option B - Command Line:")
    print("    python app/main.py --image path/to/image.jpg")
    
    print("\n" + "="*80)
    print("📚 For more information, see README.md")
    print("="*80 + "\n")

def main():
    """Main setup function"""
    print_header("🔥 WILDFIRE MANAGEMENT SYSTEM - QUICK START")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Setup environment file
    env_exists = setup_environment_file()
    
    # Check credentials
    has_kaggle = check_kaggle_credentials()
    has_groq = check_groq_api_key()
    
    # Install dependencies
    print_header("📦 DEPENDENCIES")
    install_dependencies()
    
    # Show next steps
    show_next_steps(has_kaggle, has_groq)
    
    # Quick action menu
    if has_kaggle:
        print("\n🚀 QUICK ACTIONS")
        print("1. Download datasets now")
        print("2. Launch dashboard")
        print("3. Exit and configure manually")
        
        choice = input("\nEnter choice (1-3) [3]: ").strip() or '3'
        
        if choice == '1':
            print("\n📥 Starting dataset download...")
            import subprocess
            subprocess.run([sys.executable, 
                          str(PROJECT_ROOT / "src" / "data_loader" / "download_datasets.py")])
        elif choice == '2':
            print("\n🚀 Launching dashboard...")
            print("   Note: You need trained models to use the dashboard")
            choice2 = input("   Continue? (y/n) [n]: ").strip().lower() or 'n'
            if choice2 == 'y':
                import subprocess
                subprocess.run([sys.executable, "-m", "streamlit", "run",
                              str(PROJECT_ROOT / "src" / "coordination" / "dashboard.py")])

if __name__ == "__main__":
    main()
