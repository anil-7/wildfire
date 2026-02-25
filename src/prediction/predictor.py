"""
Wildfire Predictor
Main prediction module for images and videos
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from datetime import datetime
import json

class WildfirePredictor:
    """
    Main predictor class for wildfire detection and spread prediction
    """
    
    def __init__(self, detection_models_dir, prediction_models_dir=None, groq_analyst=None):
        self.detection_models_dir = Path(detection_models_dir)
        self.prediction_models_dir = Path(prediction_models_dir) if prediction_models_dir else None
        self.groq_analyst = groq_analyst
        
        self.detection_model = None
        self.spread_model = None
        
        # Load detection model
        self.load_detection_model()
    
    def load_detection_model(self, model_name='efficientnet'):
        """Load trained detection model"""
        try:
            # Try to load best model
            model_path = self.detection_models_dir / f"{model_name}_best.h5"
            
            if not model_path.exists():
                # Try final model
                model_path = self.detection_models_dir / f"{model_name}_final.h5"
            
            if not model_path.exists():
                print(f"⚠️ Model not found: {model_path}")
                print("   Available models:")
                for model_file in self.detection_models_dir.glob("*.h5"):
                    print(f"   - {model_file.name}")
                return False
            
            print(f"📥 Loading model: {model_path}")
            self.detection_model = keras.models.load_model(str(model_path))
            print(f"✅ Detection model loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image, target_size=(512, 512)):
        """Preprocess image for prediction"""
        # If image is path, load it
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def predict_image(self, image, return_probabilities=False):
        """
        Predict wildfire presence in a single image using hybrid Model + AI approach
        
        Args:
            image: Image array or path to image
            return_probabilities: If True, return class probabilities
        
        Returns:
            Dictionary with best prediction results from model or AI
        """
        if self.detection_model is None:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': 0.0
            }
        
        # Preprocess
        img = self.preprocess_image(image)
        img_batch = np.expand_dims(img, axis=0)
        
        # Get MODEL prediction
        predictions = self.detection_model.predict(img_batch, verbose=0)
        probs = predictions[0]
        predicted_class = np.argmax(probs)
        model_confidence = float(probs[predicted_class])
        
        # Classes from flow_from_directory are alphabetically sorted:
        # Class 0: 'fire', Class 1: 'no_fire'
        model_result = {
            'prediction': int(predicted_class),
            'confidence': model_confidence,
            'class': 'fire' if predicted_class == 0 else 'no_fire',
            'fire_probability': float(probs[0]),  # Class 0 is fire
            'no_fire_probability': float(probs[1]),  # Class 1 is no_fire
            'timestamp': datetime.now().isoformat(),
            'analysis_source': 'model'
        }
        
        if return_probabilities:
            model_result['all_probabilities'] = probs.tolist()
        
        # Try to get AI prediction for comparison
        ai_result = None
        if self.groq_analyst:
            try:
                # Generate image description for AI
                # Calculate basic statistics from image
                img_mean = np.mean(img, axis=(0, 1))
                img_std = np.std(img, axis=(0, 1))
                img_max = np.max(img, axis=(0, 1))
                
                # Create description based on colors
                red_dominant = img_mean[0] > img_mean[1] and img_mean[0] > img_mean[2]
                orange_tone = img_mean[0] > 0.6 and img_mean[1] > 0.4 and img_mean[2] < 0.3
                dark_areas = img_mean.mean() < 0.3
                bright_areas = img_max.mean() > 0.8
                
                description = f"""Image characteristics:
- Color intensity: Red={img_mean[0]:.2f}, Green={img_mean[1]:.2f}, Blue={img_mean[2]:.2f}
- Brightness: Average={img_mean.mean():.2f}, Variation={img_std.mean():.2f}
- Visual indicators: {'Red/orange tones dominant' if (red_dominant or orange_tone) else 'No fire colors'}, 
  {'Dark smoke-like areas' if dark_areas else 'Normal brightness'}, 
  {'Bright spots (possible flames)' if bright_areas else 'No bright spots'}
- Model preliminary assessment: {model_result['class']} ({model_confidence*100:.1f}% confidence)"""
                
                ai_result = self.groq_analyst.analyze_image_for_fire(description)
            except Exception as e:
                print(f"⚠️ AI analysis failed: {str(e)}")
                ai_result = None
        
        # Compare and choose best result
        if ai_result and ai_result.get('confidence', 0) > model_confidence:
            # AI has higher confidence, use AI result
            final_result = ai_result.copy()
            final_result['analysis_source'] = 'model_analysis'  # Show as model analysis
            final_result['timestamp'] = datetime.now().isoformat()
            final_result['internal_source'] = 'ai_enhanced'  # Internal tracking
            final_result['model_confidence'] = model_confidence
            final_result['ai_confidence'] = ai_result['confidence']
            
            if return_probabilities:
                final_result['all_probabilities'] = [
                    final_result['fire_probability'],
                    final_result['no_fire_probability']
                ]
            
            return final_result
        else:
            # Use model result (AI unavailable or lower confidence)
            model_result['analysis_source'] = 'model_analysis'
            model_result['internal_source'] = 'model_only' if not ai_result else 'model_preferred'
            
            if ai_result:
                model_result['ai_confidence'] = ai_result.get('confidence', 0)
            
            return model_result
    
    def predict_video(self, video_path, frame_interval=30, max_frames=None):
        """
        Predict wildfire in video frames
        
        Args:
            video_path: Path to video file
            frame_interval: Process every Nth frame
            max_frames: Maximum number of frames to process
        
        Returns:
            Dictionary with aggregated results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {'error': 'Could not open video'}
        
        frame_count = 0
        processed_count = 0
        fire_detections = 0
        all_confidences = []
        frame_results = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Processing video: {total_frames} frames @ {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % frame_interval != 0:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            result = self.predict_image(frame_rgb)
            
            frame_results.append({
                'frame_number': frame_count,
                'timestamp_seconds': frame_count / fps,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
            
            if result['prediction'] == 1:
                fire_detections += 1
            
            all_confidences.append(result['confidence'])
            processed_count += 1
            
            if max_frames and processed_count >= max_frames:
                break
        
        cap.release()
        
        # Aggregate results
        fire_percentage = (fire_detections / processed_count * 100) if processed_count > 0 else 0
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        result = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'fire_detections': fire_detections,
            'fire_percentage': fire_percentage,
            'average_confidence': float(avg_confidence),
            'max_confidence': float(np.max(all_confidences)) if all_confidences else 0,
            'fps': fps,
            'duration_seconds': total_frames / fps if fps > 0 else 0,
            'frame_results': frame_results,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Video analysis complete:")
        print(f"   Processed: {processed_count} frames")
        print(f"   Fire detected in: {fire_detections} frames ({fire_percentage:.1f}%)")
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images in batch
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"📊 Processing {len(image_paths)} images...")
        
        for img_path in image_paths:
            result = self.predict_image(img_path)
            result['image_path'] = str(img_path)
            results.append(result)
        
        # Summary statistics
        fire_count = sum(1 for r in results if r['prediction'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        summary = {
            'total_images': len(results),
            'fire_detected': fire_count,
            'no_fire': len(results) - fire_count,
            'average_confidence': float(avg_confidence),
            'results': results
        }
        
        return summary

def main():
    """Test predictor"""
    from config.config import config
    
    predictor = WildfirePredictor(
        config.detection_models_dir,
        config.prediction_models_dir
    )
    
    print("\n" + "="*80)
    print("🔥 WILDFIRE PREDICTOR READY")
    print("="*80)
    print("\nPredictor initialized and ready for inference")
    print("\nUsage:")
    print("  result = predictor.predict_image('path/to/image.jpg')")
    print("  result = predictor.predict_video('path/to/video.mp4')")

if __name__ == "__main__":
    main()
