"""
Kaggle Dataset Downloader for Wildfire Datasets
Downloads and manages wildfire-related datasets from Kaggle
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
import subprocess
import json

class KaggleDatasetDownloader:
    def __init__(self):
        self.raw_data_dir = config.raw_data_dir
        
        # Popular wildfire datasets on Kaggle
        self.datasets = {
            "wildfire_images": [
                "phylake1337/fire-dataset",
                "elmadafri/the-wildfire-dataset",
                "brsdincer/wildfire-detection-image-data"
            ],
            "wildfire_spread": [
                "rtatman/188-million-us-wildfires",
                "capcloudcoder/us-wildfire-data-plus-other-attributes"
            ],
            "forest_fire": [
                "elikplim/forest-fires-data-set",
                "sumitm004/forest-fire-dataset"
            ]
        }
    
    def download_dataset(self, dataset_name):
        """Download a specific dataset from Kaggle"""
        try:
            # Create target directory
            target_dir = self.raw_data_dir / dataset_name.replace("/", "_")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 Downloading dataset: {dataset_name}")
            print(f"   Target directory: {target_dir}")
            
            # Download using kaggle CLI
            cmd = f"kaggle datasets download -d {dataset_name} -p {target_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully downloaded: {dataset_name}")
                return True
            else:
                print(f"❌ Error downloading {dataset_name}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Exception while downloading {dataset_name}: {str(e)}")
            return False
    
    def download_all_datasets(self):
        """Download all recommended wildfire datasets"""
        print("=" * 80)
        print("🔥 WILDFIRE DATASET DOWNLOADER")
        print("=" * 80)
        
        # Setup Kaggle credentials first
        if not config.setup_kaggle():
            print("\n❌ Cannot proceed without Kaggle credentials")
            return
        
        total = 0
        successful = 0
        
        for category, dataset_list in self.datasets.items():
            print(f"\n📂 Category: {category.upper()}")
            print("-" * 80)
            
            for dataset in dataset_list:
                total += 1
                if self.download_dataset(dataset):
                    successful += 1
                print()
        
        print("=" * 80)
        print(f"📊 Download Summary: {successful}/{total} datasets downloaded successfully")
        print("=" * 80)
    
    def download_specific_datasets(self, dataset_names):
        """Download specific datasets by name"""
        config.setup_kaggle()
        
        for dataset in dataset_names:
            self.download_dataset(dataset)
    
    def list_available_datasets(self):
        """List all available datasets"""
        print("\n🔥 Available Wildfire Datasets:")
        print("=" * 80)
        
        for category, dataset_list in self.datasets.items():
            print(f"\n📂 {category.upper()}:")
            for i, dataset in enumerate(dataset_list, 1):
                print(f"   {i}. {dataset}")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function"""
    downloader = KaggleDatasetDownloader()
    
    print("\n" + "=" * 80)
    print("🔥 WILDFIRE AI DATASET DOWNLOADER")
    print("=" * 80)
    
    downloader.list_available_datasets()
    
    print("\nOptions:")
    print("1. Download all datasets (recommended)")
    print("2. Download wildfire images only")
    print("3. Download wildfire spread data only")
    print("4. List datasets only")
    
    choice = input("\nEnter your choice (1-4) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        downloader.download_all_datasets()
    elif choice == "2":
        for dataset in downloader.datasets["wildfire_images"]:
            downloader.download_dataset(dataset)
    elif choice == "3":
        for dataset in downloader.datasets["wildfire_spread"]:
            downloader.download_dataset(dataset)
    elif choice == "4":
        print("\n✅ Dataset list displayed above")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
