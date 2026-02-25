"""
Configuration Manager for Wildfire Management System
Handles loading and managing all configuration settings
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        
        # Load environment variables
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Paths
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.models_dir = self.base_dir / "models"
        self.detection_models_dir = self.models_dir / "detection"
        self.prediction_models_dir = self.models_dir / "prediction"
        self.outputs_dir = self.base_dir / "outputs"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Kaggle configuration
        self.kaggle_config_path = self.config_dir / "kaggle.json"
        
        # Groq API
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        # Model parameters
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", "512"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.epochs = int(os.getenv("EPOCHS", "50"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        
        # Prediction thresholds
        self.fire_detection_threshold = float(os.getenv("FIRE_DETECTION_THRESHOLD", "0.7"))
        self.spread_prediction_confidence = float(os.getenv("SPREAD_PREDICTION_CONFIDENCE", "0.8"))
        
        # Emergency coordination
        self.alert_email = os.getenv("ALERT_EMAIL", "emergency@firefighting.com")
        self.dashboard_port = int(os.getenv("DASHBOARD_PORT", "8501"))
        self.api_port = int(os.getenv("API_PORT", "5000"))
        
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.detection_models_dir,
            self.prediction_models_dir,
            self.outputs_dir,
            self.outputs_dir / "visualizations",
            self.outputs_dir / "predictions",
            self.outputs_dir / "reports"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def setup_kaggle(self):
        """Setup Kaggle API credentials"""
        if not self.kaggle_config_path.exists():
            print(f"⚠️ Kaggle credentials not found at {self.kaggle_config_path}")
            print("Please place your kaggle.json file in the config/ directory")
            return False
        
        # Set Kaggle credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        target_path = kaggle_dir / "kaggle.json"
        if not target_path.exists():
            import shutil
            shutil.copy(self.kaggle_config_path, target_path)
            
            # Set permissions (Unix-like systems)
            if os.name != 'nt':  # Not Windows
                os.chmod(target_path, 0o600)
        
        print("✅ Kaggle credentials configured successfully")
        return True
    
    def validate_groq_api(self):
        """Validate Groq API key is configured"""
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            print("⚠️ Groq API key not configured in .env file")
            return False
        print("✅ Groq API key configured")
        return True
    
    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            "image_size": (self.max_image_size, self.max_image_size),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "fire_threshold": self.fire_detection_threshold,
            "spread_confidence": self.spread_prediction_confidence
        }

# Global config instance
config = Config()
