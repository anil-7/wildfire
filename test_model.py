"""Quick test to verify model predictions"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.prediction.predictor import WildfirePredictor
from config.config import config

# Initialize predictor
print("Loading predictor...")
predictor = WildfirePredictor(config.detection_models_dir)

# Test images
test_fire_images = list(Path("data/processed/test/fire").glob("*.png"))[:3]
test_no_fire_images = list(Path("data/processed/test/no_fire").glob("*.png"))[:3]

print("\n" + "="*60)
print("Testing FIRE images:")
print("="*60)

for img_path in test_fire_images:
    result = predictor.predict_image(str(img_path), return_probabilities=True)
    print(f"\nImage: {img_path.name}")
    print(f"  Prediction: {result['prediction']} ({result['class']})")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Fire Prob:  {result['fire_probability']*100:.2f}%")
    print(f"  NoFire Prob: {result['no_fire_probability']*100:.2f}%")
    if 'all_probabilities' in result:
        print(f"  Raw output: {result['all_probabilities']}")

print("\n" + "="*60)
print("Testing NO FIRE images:")
print("="*60)

for img_path in test_no_fire_images:
    result = predictor.predict_image(str(img_path), return_probabilities=True)
    print(f"\nImage: {img_path.name}")
    print(f"  Prediction: {result['prediction']} ({result['class']})")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Fire Prob:  {result['fire_probability']*100:.2f}%")
    print(f"  NoFire Prob: {result['no_fire_probability']*100:.2f}%")
    if 'all_probabilities' in result:
        print(f"  Raw output: {result['all_probabilities']}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
