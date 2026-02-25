"""
Data Preprocessing Module
Handles image preprocessing, augmentation, and dataset preparation
"""

import cv2
import numpy as np
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

class WildfireDataPreprocessor:
    def __init__(self, raw_data_dir, processed_data_dir, image_size=(512, 512)):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.image_size = image_size
        
        # Create processed data structure
        self.train_dir = self.processed_data_dir / "train"
        self.val_dir = self.processed_data_dir / "val"
        self.test_dir = self.processed_data_dir / "test"
        
        for split in [self.train_dir, self.val_dir, self.test_dir]:
            (split / "fire").mkdir(parents=True, exist_ok=True)
            (split / "no_fire").mkdir(parents=True, exist_ok=True)
    
    def preprocess_image(self, image_path, enhance=True):
        """Preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Resize
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
            
            if enhance:
                # Enhance contrast using CLAHE
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def augment_image(self, image):
        """Apply data augmentation"""
        augmented_images = [image]
        
        # Horizontal flip
        augmented_images.append(cv2.flip(image, 1))
        
        # Rotation
        angles = [90, 180, 270]
        for angle in angles:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented_images.append(rotated)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented_images.extend([bright, dark])
        
        return augmented_images
    
    def organize_dataset(self, source_dirs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Organize raw images into train/val/test splits"""
        print("\n📊 Organizing dataset into train/val/test splits...")
        
        all_fire_images = []
        all_no_fire_images = []
        
        # Collect all images
        for source_dir in source_dirs:
            source_path = self.raw_data_dir / source_dir
            if not source_path.exists():
                print(f"⚠️ Directory not found: {source_path}")
                continue
            
            print(f"📂 Processing: {source_dir}")
            
            # Look for fire and no-fire images
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        file_path = Path(root) / file
                        path_str = str(file_path).lower()
                        
                        # Classify based on folder structure first, then filename
                        parent_folder = Path(root).name.lower()
                        
                        # Check if in fire folder
                        if any(keyword in parent_folder 
                               for keyword in ['fire_images', 'fire', 'wildfire', 'smoke', 'flame', 'burning']):
                            # But exclude if in non-fire subfolder
                            if not any(keyword in parent_folder 
                                      for keyword in ['non_fire', 'no_fire', 'nofire', 'normal']):
                                all_fire_images.append(file_path)
                                continue
                        
                        # Check if in no-fire folder
                        if any(keyword in parent_folder 
                               for keyword in ['non_fire', 'no_fire', 'nofire', 'normal', 'landscape', 
                                              'nature', 'forest_images', 'regular']):
                            all_no_fire_images.append(file_path)
                            continue
                        
                        # Fallback to filename classification
                        is_fire = any(keyword in path_str 
                                     for keyword in ['fire', 'wildfire', 'smoke', 'flame', 'burning'])
                        is_no_fire = any(keyword in path_str 
                                        for keyword in ['nofire', 'no_fire', 'normal', 'landscape'])
                        
                        if is_fire and not is_no_fire:
                            all_fire_images.append(file_path)
                        elif is_no_fire:
                            all_no_fire_images.append(file_path)
        
        print(f"\n📈 Dataset statistics:")
        print(f"   Fire images: {len(all_fire_images)}")
        print(f"   No-fire images: {len(all_no_fire_images)}")
        
        # Split datasets
        fire_train, fire_temp = train_test_split(all_fire_images, 
                                                   test_size=(1-train_ratio), 
                                                   random_state=42)
        fire_val, fire_test = train_test_split(fire_temp, 
                                                 test_size=(test_ratio/(test_ratio+val_ratio)),
                                                 random_state=42)
        
        no_fire_train, no_fire_temp = train_test_split(all_no_fire_images,
                                                         test_size=(1-train_ratio),
                                                         random_state=42)
        no_fire_val, no_fire_test = train_test_split(no_fire_temp,
                                                       test_size=(test_ratio/(test_ratio+val_ratio)),
                                                       random_state=42)
        
        # Process and save images
        splits = {
            'train': {'fire': fire_train, 'no_fire': no_fire_train},
            'val': {'fire': fire_val, 'no_fire': no_fire_val},
            'test': {'fire': fire_test, 'no_fire': no_fire_test}
        }
        
        dataset_info = {}
        
        for split_name, categories in splits.items():
            split_dir = getattr(self, f"{split_name}_dir")
            dataset_info[split_name] = {}
            
            print(f"\n📁 Processing {split_name} split...")
            
            for category, images in categories.items():
                category_dir = split_dir / category
                saved_count = 0
                
                for i, img_path in enumerate(tqdm(images, desc=f"  {category}")):
                    processed_img = self.preprocess_image(img_path)
                    
                    if processed_img is not None:
                        # Save original processed image
                        output_path = category_dir / f"{category}_{i:05d}.png"
                        cv2.imwrite(str(output_path), (processed_img * 255).astype(np.uint8))
                        saved_count += 1
                
                dataset_info[split_name][category] = saved_count
                print(f"   ✅ {category}: {saved_count} images")
        
        # Save dataset info
        info_path = self.processed_data_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print(f"\n✅ Dataset organization complete!")
        print(f"📄 Dataset info saved to: {info_path}")
        
        return dataset_info

def main():
    """Main preprocessing function"""
    from config.config import config
    
    preprocessor = WildfireDataPreprocessor(
        config.raw_data_dir,
        config.processed_data_dir,
        image_size=(config.max_image_size, config.max_image_size)
    )
    
    # List available datasets
    print("\n🔍 Scanning for downloaded datasets...")
    available_datasets = []
    
    for item in config.raw_data_dir.iterdir():
        if item.is_dir():
            available_datasets.append(item.name)
            print(f"   ✓ {item.name}")
    
    if not available_datasets:
        print("\n❌ No datasets found in raw data directory!")
        print("   Please run download_datasets.py first")
        return
    
    print(f"\n📊 Found {len(available_datasets)} dataset(s)")
    
    # Organize dataset
    dataset_info = preprocessor.organize_dataset(available_datasets)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.config import config
    main()
