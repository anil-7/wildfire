"""
Groq AI Integration
Provides intelligent insights and recommendations using Groq API
"""

import os
from groq import Groq
import json
from datetime import datetime

class GroqWildfireAnalyst:
    """
    Integrates Groq AI for intelligent wildfire analysis and recommendations
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            print("⚠️ Warning: Groq API key not configured")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                print("✅ Groq AI initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing Groq: {str(e)}")
                self.client = None
    
    def analyze_detection(self, detection_result, image_metadata=None):
        """
        Analyze wildfire detection result and provide insights
        
        Args:
            detection_result: Dict with prediction, confidence, etc.
            image_metadata: Optional metadata about the image
        
        Returns:
            Dict with AI-generated insights and recommendations
        """
        if not self.client:
            return self._get_fallback_analysis(detection_result)
        
        try:
            # Prepare context
            context = self._prepare_detection_context(detection_result, image_metadata)
            
            # Create prompt
            prompt = f"""You are an expert wildfire analyst AI. Analyze the following wildfire detection data and provide actionable insights.

Detection Data:
{json.dumps(context, indent=2)}

Please provide:
1. Risk Assessment: Evaluate the immediate threat level
2. Recommended Actions: Specific steps for emergency response teams
3. Resource Allocation: Suggest optimal deployment of firefighting resources
4. Safety Measures: Critical safety recommendations for nearby populations
5. Monitoring Strategy: What to monitor in the coming hours

Provide your analysis in a structured, clear format."""

            # Get AI response
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert wildfire management AI assistant with deep knowledge of fire behavior, emergency response, and disaster management."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = chat_completion.choices[0].message.content
            
            return {
                'status': 'success',
                'analysis': analysis,
                'model': 'groq-llama-3.3-70b',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in Groq analysis: {str(e)}")
            return self._get_fallback_analysis(detection_result)
    
    def analyze_spread_prediction(self, spread_analysis, environmental_data=None):
        """
        Analyze fire spread prediction and provide strategic recommendations
        
        Args:
            spread_analysis: Dict with spread prediction results
            environmental_data: Optional environmental conditions data
        
        Returns:
            Dict with AI-generated strategic insights
        """
        if not self.client:
            return self._get_fallback_spread_analysis(spread_analysis)
        
        try:
            context = self._prepare_spread_context(spread_analysis, environmental_data)
            
            prompt = f"""You are an expert wildfire spread analyst. Analyze the following fire spread prediction data and provide strategic recommendations.

Spread Prediction Data:
{json.dumps(context, indent=2)}

Please provide:
1. Spread Pattern Analysis: Interpret the predicted spread pattern
2. Containment Strategy: Recommend optimal containment approach
3. Evacuation Planning: Suggest evacuation priorities and routes
4. Critical Timeframes: Identify time-sensitive action windows
5. Resource Positioning: Where to position firefighting resources

Provide actionable, time-critical recommendations."""

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in wildfire behavior modeling and emergency management strategy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = chat_completion.choices[0].message.content
            
            return {
                'status': 'success',
                'analysis': analysis,
                'model': 'groq-llama-3.3-70b',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in Groq analysis: {str(e)}")
            return self._get_fallback_spread_analysis(spread_analysis)
    
    def generate_emergency_report(self, detection_data, spread_data, historical_context=None):
        """
        Generate comprehensive emergency coordination report
        
        Args:
            detection_data: Current detection information
            spread_data: Spread prediction information
            historical_context: Optional historical fire data
        
        Returns:
            Dict with comprehensive emergency report
        """
        if not self.client:
            return self._get_fallback_emergency_report(detection_data, spread_data)
        
        try:
            prompt = f"""Generate a comprehensive emergency coordination report for disaster management agencies.

Current Detection:
{json.dumps(detection_data, indent=2)}

Spread Prediction:
{json.dumps(spread_data, indent=2)}

Create a detailed report including:
1. EXECUTIVE SUMMARY: Quick overview for decision makers
2. IMMEDIATE THREATS: Critical risks requiring immediate action
3. RESOURCE REQUIREMENTS: Estimated personnel, equipment, water resources
4. COORDINATION PLAN: How different agencies should coordinate
5. COMMUNICATION STRATEGY: Key messages for public and media
6. SUCCESS METRICS: How to measure response effectiveness

Format as a professional emergency management report."""

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior emergency management coordinator specializing in wildfire disaster response."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=2000
            )
            
            report = chat_completion.choices[0].message.content
            
            return {
                'status': 'success',
                'report': report,
                'model': 'groq-llama-3.3-70b',
                'timestamp': datetime.now().isoformat(),
                'report_id': f"WILDFIRE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            }
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            return self._get_fallback_emergency_report(detection_data, spread_data)
    
    def analyze_image_for_fire(self, image_description=None):
        """
        Use Groq AI to analyze image for fire presence
        
        Args:
            image_description: Description of what's visible in the image
        
        Returns:
            Dict with AI prediction and confidence
        """
        if not self.client:
            return None
        
        try:
            # Create prompt for fire detection
            prompt = f"""You are an expert wildfire detection AI. Based on the following image analysis, determine if there is a wildfire present.

Image Analysis: {image_description if image_description else "Visual analysis of potential wildfire scene"}

Analyze for:
- Presence of flames, smoke, or fire
- Environmental indicators (charred vegetation, ash, burnt areas)
- Color patterns indicating heat or combustion
- Smoke patterns and density

Respond ONLY with a JSON object in this exact format:
{{
    "fire_detected": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Be precise. If uncertain, set confidence below 0.7."""

            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert wildfire detection system. Analyze images and provide accurate fire detection results in JSON format only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if "```json" in ai_response:
                ai_response = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                ai_response = ai_response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(ai_response)
            
            # Convert to standard format
            prediction = 0 if result.get('fire_detected', False) else 1  # 0=fire, 1=no_fire
            confidence = float(result.get('confidence', 0.5))
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'fire_probability': confidence if prediction == 0 else 1 - confidence,
                'no_fire_probability': 1 - confidence if prediction == 0 else confidence,
                'reasoning': result.get('reasoning', ''),
                'source': 'groq_ai',
                'class': 'fire' if prediction == 0 else 'no_fire'
            }
            
        except Exception as e:
            print(f"⚠️ AI image analysis failed: {str(e)}")
            return None
    
    def _prepare_detection_context(self, detection_result, metadata):
        """Prepare context for detection analysis"""
        context = {
            'fire_detected': detection_result.get('prediction') == 1,
            'confidence': detection_result.get('confidence', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            context.update(metadata)
        
        return context
    
    def _prepare_spread_context(self, spread_analysis, environmental_data):
        """Prepare context for spread analysis"""
        context = {
            'risk_level': spread_analysis.get('risk_level', 'Unknown'),
            'spread_percentage': spread_analysis.get('spread_percentage', 0),
            'spread_area': spread_analysis.get('spread_area_pixels', 0),
            'critical_zones': len(spread_analysis.get('critical_zones', [])),
            'recommendation': spread_analysis.get('recommended_action', 'None')
        }
        
        if environmental_data:
            context['environmental_conditions'] = environmental_data
        
        return context
    
    def _get_fallback_analysis(self, detection_result):
        """Fallback analysis when Groq is not available"""
        confidence = detection_result.get('confidence', 0)
        is_fire = detection_result.get('prediction') == 1
        
        if is_fire and confidence > 0.8:
            analysis = """FALLBACK ANALYSIS (Groq API not configured)
            
Risk Assessment: HIGH - Fire detected with high confidence
Recommended Actions:
- Immediately alert local fire departments
- Deploy initial response teams
- Begin evacuation preparation for at-risk areas

Resource Allocation:
- Position firefighting units near detected area
- Prepare water resources and equipment
- Alert nearby hospitals and emergency services

Safety Measures:
- Issue evacuation advisory for immediate area
- Set up emergency shelters
- Establish communication channels

Monitoring Strategy:
- Continuous monitoring of fire progression
- Weather condition tracking
- Regular updates to command center"""
        elif is_fire:
            analysis = """FALLBACK ANALYSIS (Groq API not configured)
            
Risk Assessment: MODERATE - Possible fire detected
Recommended Actions:
- Verify detection with ground teams or aerial surveillance
- Position rapid response units on standby
- Monitor situation closely

Resource Allocation:
- Pre-position firefighting resources
- Alert emergency services
- Prepare evacuation routes

Safety Measures:
- Issue fire watch advisory
- Inform nearby residents
- Test communication systems

Monitoring Strategy:
- Enhanced monitoring of area
- Real-time updates every 15 minutes"""
        else:
            analysis = """FALLBACK ANALYSIS (Groq API not configured)
            
Risk Assessment: LOW - No fire detected
Recommended Actions:
- Continue routine monitoring
- Maintain readiness protocols

Resource Allocation:
- Standard resource positioning
- Regular equipment maintenance

Safety Measures:
- Normal operations
- Public awareness campaigns

Monitoring Strategy:
- Routine surveillance schedule"""
        
        return {
            'status': 'fallback',
            'analysis': analysis,
            'model': 'rule-based-fallback',
            'timestamp': datetime.now().isoformat(),
            'note': 'Configure Groq API key for AI-powered insights'
        }
    
    def _get_fallback_spread_analysis(self, spread_analysis):
        """Fallback for spread analysis"""
        risk = spread_analysis.get('risk_level', 'Unknown')
        
        return {
            'status': 'fallback',
            'analysis': f"""FALLBACK SPREAD ANALYSIS
            
Risk Level: {risk}
Spread Area: {spread_analysis.get('spread_percentage', 0):.1f}%

Basic Recommendations:
- Monitor fire spread closely
- Position resources based on predicted zones
- Maintain evacuation readiness
- Regular status updates to command

Note: Configure Groq API for detailed AI-powered analysis.""",
            'model': 'rule-based-fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_fallback_emergency_report(self, detection_data, spread_data):
        """Fallback emergency report"""
        return {
            'status': 'fallback',
            'report': """EMERGENCY COORDINATION REPORT (Fallback Mode)
            
EXECUTIVE SUMMARY:
Automated wildfire detection and analysis system active.
Configure Groq API for comprehensive AI-powered reports.

IMMEDIATE ACTIONS:
1. Verify all detections with ground teams
2. Follow standard emergency protocols
3. Maintain communication with all agencies

CONTACT:
Emergency Operations Center
24/7 Hotline: [Configure in settings]

Note: This is a basic template. Configure Groq AI for detailed reports.""",
            'model': 'fallback-template',
            'timestamp': datetime.now().isoformat(),
            'report_id': f"WILDFIRE-FALLBACK-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }
