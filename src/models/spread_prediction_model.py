"""
Wildfire Spread Prediction Model
Uses spatio-temporal deep learning to forecast fire spread patterns
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class WildfireSpreadPredictor:
    """
    Predictive model for wildfire spread patterns
    Uses ConvLSTM and attention mechanisms for temporal-spatial prediction
    """
    
    def __init__(self, input_shape=(5, 256, 256, 3), num_forecast_steps=3):
        """
        Args:
            input_shape: (timesteps, height, width, channels)
            num_forecast_steps: Number of future timesteps to predict
        """
        self.input_shape = input_shape
        self.num_forecast_steps = num_forecast_steps
    
    def create_convlstm_model(self, name="convlstm_spread"):
        """Create ConvLSTM model for spread prediction"""
        model = models.Sequential([
            layers.ConvLSTM2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True,
                input_shape=self.input_shape
            ),
            layers.BatchNormalization(),
            
            layers.ConvLSTM2D(
                filters=128,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True
            ),
            layers.BatchNormalization(),
            
            layers.ConvLSTM2D(
                filters=256,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=False
            ),
            layers.BatchNormalization(),
            
            # Decoder for prediction
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Output: Predicted fire spread mask
            layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')
        ], name=name)
        
        return model
    
    def create_unet_temporal(self, name="unet_temporal"):
        """Create U-Net based temporal model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Flatten time dimension for initial processing
        x = layers.TimeDistributed(layers.Conv2D(64, 3, activation='relu', padding='same'))(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        
        # ConvLSTM for temporal modeling
        x = layers.ConvLSTM2D(64, 3, padding='same', return_sequences=False)(x)
        
        # U-Net encoder
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
        
        # U-Net decoder
        u3 = layers.UpSampling2D((2, 2))(c4)
        u3 = layers.Concatenate()([u3, c3])
        c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u3)
        c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
        
        u2 = layers.UpSampling2D((2, 2))(c5)
        u2 = layers.Concatenate()([u2, c2])
        c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
        c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
        
        u1 = layers.UpSampling2D((2, 2))(c6)
        u1 = layers.Concatenate()([u1, c1])
        c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
        c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
        
        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def create_attention_spread_model(self, name="attention_spread"):
        """Create attention-based spread prediction model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Process temporal sequence
        x = layers.ConvLSTM2D(64, 3, padding='same', return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        
        # Spatial attention
        attention = layers.ConvLSTM2D(1, 3, padding='same', return_sequences=True, 
                                      activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        # Continue processing
        x = layers.ConvLSTM2D(128, 3, padding='same', return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(256, 3, padding='same', return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        
        # Prediction head
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile model with appropriate loss and metrics"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.MeanIoU(num_classes=2, name='iou')
            ]
        )
        
        return model
    
    def get_callbacks(self, model_name, checkpoint_dir):
        """Get training callbacks"""
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_dir}/{model_name}_best.h5",
                monitor='val_iou',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks

class SpreadAnalyzer:
    """
    Analyzes predicted spread patterns and provides insights
    """
    
    def __init__(self):
        self.risk_levels = {
            0.0: "No Risk",
            0.3: "Low Risk",
            0.5: "Moderate Risk",
            0.7: "High Risk",
            0.9: "Critical Risk"
        }
    
    def analyze_spread_prediction(self, prediction_mask):
        """Analyze the prediction mask and provide risk assessment"""
        # Calculate spread metrics
        spread_area = np.sum(prediction_mask > 0.5)
        total_area = prediction_mask.size
        spread_percentage = (spread_area / total_area) * 100
        
        max_intensity = np.max(prediction_mask)
        avg_intensity = np.mean(prediction_mask[prediction_mask > 0.5]) if spread_area > 0 else 0
        
        # Determine risk level
        risk_level = self._get_risk_level(max_intensity)
        
        # Find critical zones (high probability areas)
        critical_zones = self._find_critical_zones(prediction_mask)
        
        analysis = {
            'spread_percentage': spread_percentage,
            'spread_area_pixels': int(spread_area),
            'max_intensity': float(max_intensity),
            'avg_intensity': float(avg_intensity),
            'risk_level': risk_level,
            'critical_zones': critical_zones,
            'recommended_action': self._get_recommendation(risk_level, spread_percentage)
        }
        
        return analysis
    
    def _get_risk_level(self, intensity):
        """Determine risk level based on intensity"""
        for threshold, level in sorted(self.risk_levels.items(), reverse=True):
            if intensity >= threshold:
                return level
        return "No Risk"
    
    def _find_critical_zones(self, prediction_mask, threshold=0.7):
        """Find areas with high spread probability"""
        import cv2
        
        # Threshold the mask
        binary_mask = (prediction_mask > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                zones.append({
                    'zone_id': i + 1,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': int(area)
                })
        
        return zones
    
    def _get_recommendation(self, risk_level, spread_percentage):
        """Get action recommendation based on risk"""
        if risk_level == "Critical Risk":
            return "IMMEDIATE EVACUATION REQUIRED. Deploy all available resources."
        elif risk_level == "High Risk":
            return "High priority response needed. Prepare evacuation routes."
        elif risk_level == "Moderate Risk":
            return "Monitor closely. Position firefighting teams strategically."
        elif risk_level == "Low Risk":
            return "Continue monitoring. Maintain readiness."
        else:
            return "No immediate action required. Regular surveillance."
