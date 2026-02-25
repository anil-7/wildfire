"""
Emergency Coordination Dashboard
Streamlit-based dashboard for real-time wildfire management
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime

from config.config import config
from src.prediction.predictor import WildfirePredictor
from src.groq_integration.groq_analyst import GroqWildfireAnalyst
from src.visualization.visualizer import WildfireVisualizer

# Page configuration
st.set_page_config(
    page_title="🔥 Wildfire Management System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4b4b;
    }
    .alert-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    """Main dashboard application"""
    
    def __init__(self):
        self.predictor = None
        self.groq_analyst = None
        self.visualizer = WildfireVisualizer(config.outputs_dir / "visualizations")
        
        # Initialize session state
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'alert_count' not in st.session_state:
            st.session_state.alert_count = 0
    
    def load_models(self):
        """Load trained models"""
        if self.predictor is None:
            with st.spinner("Loading AI models..."):
                try:
                    self.predictor = WildfirePredictor(
                        config.detection_models_dir,
                        config.prediction_models_dir
                    )
                    st.success("✅ AI models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
                    st.info("Please train models first using train_detection_model.py")
                    return False
        return True
    
    def initialize_groq(self):
        """Initialize Groq AI"""
        if self.groq_analyst is None:
            try:
                self.groq_analyst = GroqWildfireAnalyst()
            except Exception as e:
                st.warning(f"⚠️ Groq AI not available: {str(e)}")
                st.info("Add GROQ_API_KEY to .env file for AI-powered insights")
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🔥 AI-Integrated Smart Wildfire Management System</h1>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(st.session_state.detection_history))
        with col2:
            fire_count = sum(1 for d in st.session_state.detection_history 
                           if d.get('prediction') == 1)
            st.metric("Fire Alerts", fire_count)
        with col3:
            st.metric("Active Alerts", st.session_state.alert_count)
        with col4:
            st.metric("System Status", "🟢 ONLINE")
    
    def render_sidebar(self):
        """Render sidebar"""
        st.sidebar.title("🎛️ Control Panel")
        
        mode = st.sidebar.radio(
            "Select Mode",
            ["Early Detection", "Spread Prediction", "Emergency Coordination", "Analytics"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Settings")
        
        confidence_threshold = st.sidebar.slider(
            "Detection Threshold",
            0.0, 1.0, config.fire_detection_threshold, 0.05
        )
        
        enable_groq = st.sidebar.checkbox("Enable AI Insights (Groq)", value=True)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 System Info")
        st.sidebar.info(f"""
**Model**: Hybrid Ensemble CNN
**Image Size**: {config.max_image_size}px
**Confidence**: {confidence_threshold*100:.0f}%
**Status**: Active
        """)
        
        return mode, confidence_threshold, enable_groq
    
    def early_detection_mode(self, confidence_threshold, enable_groq):
        """Early detection interface"""
        st.header("🔍 Early Wildfire Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Image or Video")
            
            upload_type = st.radio("Input Type", ["Image", "Video"], horizontal=True)
            
            if upload_type == "Image":
                uploaded_file = st.file_uploader(
                    "Choose an image...",
                    type=['jpg', 'jpeg', 'png', 'bmp']
                )
                
                if uploaded_file is not None:
                    # Display image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("🔥 Analyze for Wildfire", type="primary"):
                        if not self.load_models():
                            return
                        
                        with st.spinner("Analyzing image..."):
                            # Convert to array
                            img_array = np.array(image)
                            
                            # Predict
                            result = self.predictor.predict_image(img_array)
                            
                            # Store in history
                            st.session_state.detection_history.append(result)
                            if result['prediction'] == 1:
                                st.session_state.alert_count += 1
                            
                            # Display results in col2
                            self.display_detection_result(result, enable_groq, col2)
            
            else:  # Video
                st.info("Video analysis coming soon! Use image mode for now.")
        
        with col2:
            st.subheader("Detection Results")
            st.write("Upload an image to see results")
    
    def display_detection_result(self, result, enable_groq, container):
        """Display detection results"""
        with container:
            st.subheader("🎯 Detection Results")
            
            # Prediction
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == 1:
                st.markdown(f'<div class="alert-high"><h3>🔥 FIRE DETECTED</h3><p>Confidence: {confidence*100:.2f}%</p></div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low"><h3>✅ NO FIRE</h3><p>Confidence: {confidence*100:.2f}%</p></div>',
                          unsafe_allow_html=True)
            
            # Show detailed metrics
            st.markdown("#### Detailed Metrics")
            for key, value in result.items():
                if key not in ['prediction', 'confidence', 'timestamp']:
                    st.write(f"**{key}**: {value}")
            
            # AI Insights
            if enable_groq and prediction == 1:
                st.markdown("---")
                st.markdown("#### 🤖 AI-Powered Insights")
                
                self.initialize_groq()
                if self.groq_analyst:
                    with st.spinner("Generating AI insights..."):
                        analysis = self.groq_analyst.analyze_detection(result)
                        st.markdown(analysis.get('analysis', 'No analysis available'))
                else:
                    st.warning("Groq AI not configured")
    
    def spread_prediction_mode(self):
        """Spread prediction interface"""
        st.header("📈 Fire Spread Prediction")
        st.info("Upload a sequence of images to predict fire spread patterns")
        
        st.write("This feature requires temporal data. Upload multiple images:")
        
        uploaded_files = st.file_uploader(
            "Upload 3-5 sequential images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) >= 3:
            st.success(f"Uploaded {len(uploaded_files)} images")
            
            # Display images
            cols = st.columns(len(uploaded_files))
            for idx, (col, file) in enumerate(zip(cols, uploaded_files)):
                image = Image.open(file)
                col.image(image, caption=f"Time {idx+1}", use_column_width=True)
            
            if st.button("🔮 Predict Spread Pattern", type="primary"):
                st.info("Spread prediction model integration coming soon!")
    
    def emergency_coordination_mode(self, enable_groq):
        """Emergency coordination interface"""
        st.header("🚨 Emergency Coordination Tools")
        
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Generate Report", "📞 Contacts"])
        
        with tab1:
            # Recent detections
            st.subheader("Recent Fire Detections")
            
            if st.session_state.detection_history:
                fire_detections = [d for d in st.session_state.detection_history 
                                  if d.get('prediction') == 1]
                
                if fire_detections:
                    for i, detection in enumerate(reversed(fire_detections[-5:])):
                        with st.expander(f"🔥 Alert #{len(fire_detections)-i} - {detection.get('timestamp', 'Unknown')}"):
                            st.json(detection)
                else:
                    st.success("✅ No active fire alerts")
            else:
                st.info("No detections yet")
        
        with tab2:
            st.subheader("📄 Generate Emergency Report")
            
            if st.button("Generate Comprehensive Report", type="primary"):
                if not st.session_state.detection_history:
                    st.warning("No detection data available")
                    return
                
                self.initialize_groq()
                
                latest = st.session_state.detection_history[-1]
                
                with st.spinner("Generating AI-powered emergency report..."):
                    if self.groq_analyst:
                        report = self.groq_analyst.generate_emergency_report(
                            latest, {}, None
                        )
                        
                        st.markdown("### Emergency Coordination Report")
                        st.markdown(report.get('report', 'Report generation failed'))
                        
                        # Download button
                        st.download_button(
                            "📥 Download Report",
                            report.get('report', ''),
                            file_name=f"emergency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Groq AI not configured. Cannot generate detailed report.")
        
        with tab3:
            st.subheader("📞 Emergency Contacts")
            st.markdown("""
#### Fire Department
- **Emergency**: 911
- **Non-Emergency**: (555) 123-4567

#### Emergency Operations Center
- **Hotline**: (555) 987-6543
- **Email**: emergency@wildfire.gov

#### Disaster Management Agency
- **Coordination**: (555) 246-8135
- **Email**: disaster@management.gov

#### Public Information
- **Media Hotline**: (555) 111-2222
            """)
    
    def analytics_mode(self):
        """Analytics and reporting interface"""
        st.header("📊 Analytics & Reports")
        
        if not st.session_state.detection_history:
            st.info("No data available yet. Start detecting fires to see analytics.")
            return
        
        # Statistics
        total = len(st.session_state.detection_history)
        fire_count = sum(1 for d in st.session_state.detection_history 
                        if d.get('prediction') == 1)
        no_fire_count = total - fire_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyzed", total)
        with col2:
            st.metric("Fire Detected", fire_count)
        with col3:
            st.metric("No Fire", no_fire_count)
        
        # Charts
        if fire_count > 0:
            st.subheader("📈 Detection Trends")
            
            # Simple bar chart
            chart_data = {
                'Category': ['Fire', 'No Fire'],
                'Count': [fire_count, no_fire_count]
            }
            
            st.bar_chart(chart_data, x='Category', y='Count')
        
        # Export data
        st.subheader("💾 Export Data")
        if st.button("Export Detection History"):
            data_json = json.dumps(st.session_state.detection_history, indent=2)
            st.download_button(
                "Download JSON",
                data_json,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        mode, confidence_threshold, enable_groq = self.render_sidebar()
        
        if mode == "Early Detection":
            self.early_detection_mode(confidence_threshold, enable_groq)
        elif mode == "Spread Prediction":
            self.spread_prediction_mode()
        elif mode == "Emergency Coordination":
            self.emergency_coordination_mode(enable_groq)
        elif mode == "Analytics":
            self.analytics_mode()

def main():
    """Main entry point"""
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main()
