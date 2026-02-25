"""
Hybrid Wildfire Detection Model
Combines CNN with ensemble methods for maximum accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB3, ResNet50, InceptionV3, VGG16
)
import numpy as np

class HybridWildfireDetector:
    """
    Hybrid model combining multiple CNN architectures with ensemble learning
    for optimal wildfire detection accuracy
    """
    
    def __init__(self, input_shape=(512, 512, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        
    def create_efficientnet_model(self, name="efficientnet"):
        """Create EfficientNet-based model"""
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name=name)
        
        return model
    
    def create_resnet_model(self, name="resnet"):
        """Create ResNet-based model"""
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name=name)
        
        return model
    
    def create_inception_model(self, name="inception"):
        """Create Inception-based model"""
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name=name)
        
        return model
    
    def create_custom_cnn(self, name="custom_cnn"):
        """Create custom CNN architecture optimized for wildfire detection"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name=name)
        
        return model
    
    def create_attention_cnn(self, name="attention_cnn"):
        """Create CNN with attention mechanism"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convolutional blocks
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
        x = layers.Multiply()([x, attention])
        
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def create_ensemble_model(self):
        """Create ensemble model combining multiple architectures"""
        print("🔧 Creating hybrid ensemble model...")
        
        # Create all models
        self.models['efficientnet'] = self.create_efficientnet_model()
        self.models['resnet'] = self.create_resnet_model()
        self.models['inception'] = self.create_inception_model()
        self.models['custom_cnn'] = self.create_custom_cnn()
        self.models['attention_cnn'] = self.create_attention_cnn()
        
        print(f"✅ Created {len(self.models)} models for ensemble")
        return self.models
    
    def get_best_single_model(self, model_type='efficientnet'):
        """Get a single best-performing model"""
        print(f"🔧 Creating {model_type} model...")
        
        if model_type == 'efficientnet':
            model = self.create_efficientnet_model()
        elif model_type == 'resnet':
            model = self.create_resnet_model()
        elif model_type == 'inception':
            model = self.create_inception_model()
        elif model_type == 'custom_cnn':
            model = self.create_custom_cnn()
        elif model_type == 'attention_cnn':
            model = self.create_attention_cnn()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile model with optimizer and metrics"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def get_callbacks(self, model_name, checkpoint_dir):
        """Get training callbacks"""
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_dir}/{model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=f"{checkpoint_dir}/logs/{model_name}",
                histogram_freq=1
            )
        ]
        
        return callbacks

class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def predict(self, x):
        """Make ensemble prediction"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def predict_class(self, x):
        """Predict class labels"""
        predictions = self.predict(x)
        return np.argmax(predictions, axis=1)
