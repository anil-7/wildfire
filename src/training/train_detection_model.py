"""
Training Script for Wildfire Detection Models
Trains hybrid ensemble models for maximum accuracy
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

from config.config import config
from src.models.detection_model import HybridWildfireDetector, EnsemblePredictor

class ModelTrainer:
    def __init__(self):
        self.config = config
        self.data_dir = config.processed_data_dir
        self.models_dir = config.detection_models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset info
        info_path = self.data_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = None
            print("⚠️ Dataset info not found. Please preprocess data first.")
    
    def load_data(self):
        """Load preprocessed data"""
        print("\n📊 Loading dataset...")
        
        img_size = (config.max_image_size, config.max_image_size)
        batch_size = config.batch_size
        
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation/test
        val_test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'val',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {val_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        print(f"✅ Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def train_single_model(self, model_type='efficientnet'):
        """Train a single model"""
        print(f"\n{'='*80}")
        print(f"🔥 Training {model_type.upper()} Model")
        print(f"{'='*80}\n")
        
        # Load data
        train_gen, val_gen, test_gen = self.load_data()
        
        # Create model
        detector = HybridWildfireDetector(
            input_shape=(config.max_image_size, config.max_image_size, 3),
            num_classes=2
        )
        
        model = detector.get_best_single_model(model_type)
        model = detector.compile_model(model, learning_rate=config.learning_rate)
        
        # Print model summary
        print(f"\n📋 Model Summary:")
        model.summary()
        
        # Setup callbacks
        callbacks = detector.get_callbacks(model_type, str(self.models_dir))
        
        # Train model
        print(f"\n🚀 Starting training...")
        history = model.fit(
            train_gen,
            epochs=config.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\n📊 Evaluating on test set...")
        test_results = model.evaluate(test_gen, verbose=1)
        
        # Save final model
        final_model_path = self.models_dir / f"{model_type}_final.h5"
        model.save(str(final_model_path))
        print(f"✅ Model saved: {final_model_path}")
        
        # Save history
        history_path = self.models_dir / f"{model_type}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=4)
        
        # Plot training history
        self.plot_training_history(history, model_type)
        
        # Save test results
        results = {
            'model_type': model_type,
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'test_auc': float(test_results[4]),
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.models_dir / f"{model_type}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"✅ {model_type.upper()} Training Complete!")
        print(f"   Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"   Test Precision: {results['test_precision']*100:.2f}%")
        print(f"   Test Recall: {results['test_recall']*100:.2f}%")
        print(f"   AUC: {results['test_auc']:.4f}")
        print(f"{'='*80}\n")
        
        return model, history, results
    
    def train_all_models(self):
        """Train all models in the ensemble"""
        model_types = ['efficientnet', 'resnet', 'inception', 'custom_cnn', 'attention_cnn']
        
        all_results = {}
        
        for model_type in model_types:
            try:
                model, history, results = self.train_single_model(model_type)
                all_results[model_type] = results
            except Exception as e:
                print(f"❌ Error training {model_type}: {str(e)}")
                continue
        
        # Save combined results
        combined_path = self.models_dir / "all_models_results.json"
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print("\n" + "="*80)
        print("🎉 ALL MODELS TRAINING COMPLETE!")
        print("="*80)
        print("\n📊 Results Summary:")
        for model_type, results in all_results.items():
            print(f"   {model_type:20s}: Accuracy={results['test_accuracy']*100:.2f}%")
        print("="*80 + "\n")
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name.upper()} Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Train')
            axes[1, 0].plot(history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train')
            axes[1, 1].plot(history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = config.outputs_dir / 'visualizations' / f'{model_name}_training_history.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        print(f"📈 Training history plot saved: {plot_path}")
        plt.close()

def main():
    """Main training function"""
    print("\n" + "="*80)
    print("🔥 WILDFIRE DETECTION MODEL TRAINER")
    print("="*80)
    
    trainer = ModelTrainer()
    
    if trainer.dataset_info is None:
        print("\n❌ Please run preprocessing first!")
        print("   python src/preprocessing/preprocess_data.py")
        return
    
    print("\n🎯 Training Options:")
    print("1. Train single model (EfficientNet - Recommended)")
    print("2. Train all models for ensemble (Takes longer, higher accuracy)")
    print("3. Train specific model")
    
    choice = input("\nEnter your choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        trainer.train_single_model('efficientnet')
    elif choice == "2":
        trainer.train_all_models()
    elif choice == "3":
        print("\nAvailable models:")
        print("1. efficientnet")
        print("2. resnet")
        print("3. inception")
        print("4. custom_cnn")
        print("5. attention_cnn")
        model_choice = input("\nEnter model name: ").strip().lower()
        trainer.train_single_model(model_choice)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
