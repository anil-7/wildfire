"""
Complete Workflow Script
Automates the entire pipeline from data download to model training
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

import subprocess
import time

def print_step(step_num, total, description):
    """Print formatted step header"""
    print("\n" + "="*80)
    print(f"STEP {step_num}/{total}: {description}")
    print("="*80 + "\n")

def run_script(script_path, description):
    """Run a Python script and check for errors"""
    print(f"▶️  Running: {description}")
    print(f"   Script: {script_path}\n")
    
    result = subprocess.run([sys.executable, str(script_path)])
    
    if result.returncode != 0:
        print(f"\n❌ Error running {description}")
        return False
    
    print(f"\n✅ {description} completed successfully")
    return True

def main():
    """Run complete workflow"""
    print("\n" + "="*80)
    print("🔥 WILDFIRE MANAGEMENT SYSTEM - COMPLETE WORKFLOW")
    print("="*80)
    
    print("\n⚠️  This will run the complete pipeline:")
    print("   1. Download datasets from Kaggle")
    print("   2. Preprocess data")
    print("   3. Train detection model")
    print("   4. Generate visualizations")
    print("\n⏱️  This may take several hours depending on your hardware.")
    
    choice = input("\n   Continue? (y/n) [n]: ").strip().lower() or 'n'
    
    if choice != 'y':
        print("\n❌ Workflow cancelled")
        return
    
    start_time = time.time()
    
    # Step 1: Download datasets
    print_step(1, 3, "Download Wildfire Datasets from Kaggle")
    if not run_script(
        PROJECT_ROOT / "src" / "data_loader" / "download_datasets.py",
        "Dataset Download"
    ):
        print("\n⚠️  You can continue without downloading or fix Kaggle credentials")
        choice = input("   Continue anyway? (y/n) [n]: ").strip().lower() or 'n'
        if choice != 'y':
            return
    
    # Step 2: Preprocess data
    print_step(2, 3, "Preprocess and Organize Data")
    if not run_script(
        PROJECT_ROOT / "src" / "preprocessing" / "preprocess_data.py",
        "Data Preprocessing"
    ):
        print("\n❌ Cannot continue without preprocessed data")
        return
    
    # Step 3: Train model
    print_step(3, 3, "Train Wildfire Detection Model")
    print("\n📝 Training configuration:")
    print("   You can choose to:")
    print("   1. Train single model (faster, ~1-2 hours)")
    print("   2. Train all models for ensemble (slower, ~5-8 hours, higher accuracy)")
    
    if not run_script(
        PROJECT_ROOT / "src" / "training" / "train_detection_model.py",
        "Model Training"
    ):
        print("\n❌ Training failed")
        return
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    # Success message
    print("\n" + "="*80)
    print("🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
    print("="*80)
    print(f"\n⏱️  Total time: {hours}h {minutes}m")
    print("\n✅ Your system is now ready to use!")
    print("\n📋 Next steps:")
    print("   1. Launch dashboard: python app/main.py --dashboard")
    print("   2. Or analyze image: python app/main.py --image path/to/image.jpg")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
