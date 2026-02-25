"""
Main Application Entry Point
AI-Integrated Smart Wildfire Management System
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime
import cv2

from config.config import config
from src.prediction.predictor import WildfirePredictor
from src.groq_integration.groq_analyst import GroqWildfireAnalyst
from src.visualization.visualizer import WildfireVisualizer

class WildfireManagementApp:
    """
    Main application for wildfire detection and management
    """
    
    def __init__(self, use_groq=True):
        print("\n" + "="*80)
        print("🔥 AI-INTEGRATED SMART WILDFIRE MANAGEMENT SYSTEM")
        print("="*80 + "\n")
        
        # Initialize components
        print("📦 Initializing components...")
        
        self.predictor = WildfirePredictor(
            config.detection_models_dir,
            config.prediction_models_dir
        )
        
        self.visualizer = WildfireVisualizer(
            config.outputs_dir / "visualizations"
        )
        
        self.groq_analyst = None
        if use_groq:
            try:
                self.groq_analyst = GroqWildfireAnalyst()
                print("✅ Groq AI analyst initialized")
            except Exception as e:
                print(f"⚠️ Groq AI not available: {str(e)}")
        
        self.output_dir = config.outputs_dir
        print(f"✅ Output directory: {self.output_dir}")
        print("\n" + "="*80 + "\n")
    
    def process_image(self, image_path, save_results=True):
        """
        Process a single image for wildfire detection
        
        Args:
            image_path: Path to image file
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with comprehensive results
        """
        print(f"\n📸 Processing image: {image_path}")
        print("-" * 80)
        
        # Predict
        result = self.predictor.predict_image(image_path, return_probabilities=True)
        
        # Display results
        print("\n🎯 Detection Results:")
        print(f"   Prediction: {'🔥 FIRE DETECTED' if result['prediction'] == 1 else '✅ NO FIRE'}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print(f"   Fire Probability: {result['fire_probability']*100:.2f}%")
        print(f"   No Fire Probability: {result['no_fire_probability']*100:.2f}%")
        
        # Get AI insights if fire detected
        if result['prediction'] == 1 and self.groq_analyst:
            print("\n🤖 Generating AI insights...")
            analysis = self.groq_analyst.analyze_detection(result)
            result['ai_analysis'] = analysis
            
            print("\n" + "="*80)
            print("AI ANALYSIS")
            print("="*80)
            print(analysis.get('analysis', 'No analysis available'))
            print("="*80)
        
        # Visualize
        image_name = Path(image_path).stem
        self.visualizer.visualize_prediction(
            image_path,
            result['prediction'],
            result['confidence'],
            save_name=f"{image_name}_prediction"
        )
        
        # Save results
        if save_results:
            results_file = self.output_dir / "predictions" / f"{image_name}_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=4)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        print("-" * 80)
        return result
    
    def process_video(self, video_path, frame_interval=30, save_results=True):
        """
        Process video for wildfire detection
        
        Args:
            video_path: Path to video file
            frame_interval: Process every Nth frame
            save_results: Whether to save results
        
        Returns:
            Dictionary with comprehensive results
        """
        print(f"\n📹 Processing video: {video_path}")
        print("-" * 80)
        
        # Predict
        result = self.predictor.predict_video(
            video_path,
            frame_interval=frame_interval
        )
        
        # Display results
        print("\n🎯 Video Analysis Results:")
        print(f"   Duration: {result['duration_seconds']:.1f} seconds")
        print(f"   Processed Frames: {result['processed_frames']}")
        print(f"   Fire Detected: {result['fire_detections']} frames ({result['fire_percentage']:.1f}%)")
        print(f"   Average Confidence: {result['average_confidence']*100:.2f}%")
        print(f"   Max Confidence: {result['max_confidence']*100:.2f}%")
        
        # Determine if fire is present
        fire_present = result['fire_percentage'] > 20  # More than 20% frames show fire
        
        if fire_present and self.groq_analyst:
            print("\n🤖 Generating AI insights for video...")
            analysis = self.groq_analyst.analyze_detection({
                'prediction': 1,
                'confidence': result['max_confidence'],
                'video_analysis': True,
                'fire_percentage': result['fire_percentage']
            })
            result['ai_analysis'] = analysis
            
            print("\n" + "="*80)
            print("AI ANALYSIS")
            print("="*80)
            print(analysis.get('analysis', 'No analysis available'))
            print("="*80)
        
        # Save results
        if save_results:
            video_name = Path(video_path).stem
            results_file = self.output_dir / "predictions" / f"{video_name}_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=4, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        print("-" * 80)
        return result
    
    def process_directory(self, directory_path, extensions=None):
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process
        
        Returns:
            Summary of batch processing
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        directory = Path(directory_path)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        print(f"\n📂 Processing directory: {directory}")
        print(f"   Found {len(image_paths)} images")
        print("-" * 80)
        
        results = self.predictor.predict_batch(image_paths)
        
        # Display summary
        print("\n📊 Batch Processing Summary:")
        print(f"   Total Images: {results['total_images']}")
        print(f"   Fire Detected: {results['fire_detected']}")
        print(f"   No Fire: {results['no_fire']}")
        print(f"   Average Confidence: {results['average_confidence']*100:.2f}%")
        
        # Save batch results
        batch_file = self.output_dir / "predictions" / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(batch_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n💾 Batch results saved to: {batch_file}")
        print("-" * 80)
        
        return results
    
    def generate_emergency_report(self, detection_results):
        """
        Generate emergency coordination report
        
        Args:
            detection_results: Results from detection
        
        Returns:
            Emergency report
        """
        if not self.groq_analyst:
            print("⚠️ Groq AI not available. Cannot generate detailed report.")
            return None
        
        print("\n📄 Generating Emergency Coordination Report...")
        print("-" * 80)
        
        report = self.groq_analyst.generate_emergency_report(
            detection_results,
            {},
            None
        )
        
        # Save report
        report_file = self.output_dir / "reports" / f"{report['report_id']}.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(f"EMERGENCY COORDINATION REPORT\n")
            f.write(f"Report ID: {report['report_id']}\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Model: {report['model']}\n")
            f.write(f"\n{'='*80}\n\n")
            f.write(report['report'])
        
        print(f"✅ Report saved to: {report_file}")
        print("-" * 80)
        
        return report

def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="AI-Integrated Smart Wildfire Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app/main.py --image path/to/image.jpg
  python app/main.py --video path/to/video.mp4
  python app/main.py --directory path/to/images/
  python app/main.py --image photo.jpg --no-groq
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--directory', type=str, help='Path to directory with images')
    parser.add_argument('--frame-interval', type=int, default=30, 
                       help='Process every Nth frame for video (default: 30)')
    parser.add_argument('--no-groq', action='store_true', 
                       help='Disable Groq AI insights')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch web dashboard')
    parser.add_argument('--report', action='store_true',
                       help='Generate emergency report (requires detection first)')
    
    args = parser.parse_args()
    
    # Launch dashboard if requested
    if args.dashboard:
        print("\n🚀 Launching web dashboard...")
        print("   Run: streamlit run src/coordination/dashboard.py")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "src" / "coordination" / "dashboard.py")
        ])
        return
    
    # Initialize app
    use_groq = not args.no_groq
    app = WildfireManagementApp(use_groq=use_groq)
    
    # Process based on arguments
    if args.image:
        result = app.process_image(args.image)
        
        if args.report and result['prediction'] == 1:
            app.generate_emergency_report(result)
    
    elif args.video:
        result = app.process_video(args.video, frame_interval=args.frame_interval)
        
        if args.report and result['fire_percentage'] > 20:
            app.generate_emergency_report(result)
    
    elif args.directory:
        results = app.process_directory(args.directory)
    
    else:
        parser.print_help()
        print("\n💡 TIP: Use --dashboard to launch the web interface")
        print("         python app/main.py --dashboard")

if __name__ == "__main__":
    main()
