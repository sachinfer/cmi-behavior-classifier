"""
Machine Learning model module for behavior classification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class BehaviorClassifier:
    """Main classifier for human behavior recognition"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.classes = ['walking', 'sitting', 'driving', 'standing']
        self.model_path = f"models/{model_type}_model.pkl"
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y):
        """Train the model"""
        try:
            # Ensure X is 2D
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], -1)
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Save the trained model
            self._save_model()
            
            return True
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        # Ensure X is 2D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, X):
        """Predict single sample"""
        if not self.is_trained:
            return self._generate_demo_prediction()
        
        # Ensure X is 2D
        if len(X.shape) == 3:
            X = X.reshape(1, -1)
        elif len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Create result dictionary
        result = {
            'prediction': prediction,
            'confidence': np.max(probabilities),
            'probabilities': dict(zip(self.classes, probabilities)),
            'model_used': self.model_type,
            'mode': 'real'
        }
        
        return result
    
    def _generate_demo_prediction(self):
        """Generate demo prediction when model is not trained"""
        np.random.seed(42)
        
        # Generate realistic probabilities
        probs = np.random.dirichlet([2, 2, 1, 1])
        
        # Select prediction based on highest probability
        prediction_idx = np.argmax(probs)
        prediction = self.classes[prediction_idx]
        
        result = {
            'prediction': prediction,
            'confidence': probs[prediction_idx],
            'probabilities': dict(zip(self.classes, probs)),
            'model_used': 'demo',
            'mode': 'demo'
        }
        
        return result
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if not self.is_trained:
            raise Exception("Model must be trained before evaluation")
        
        # Make predictions
        predictions, probabilities = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        cm = confusion_matrix(y, predictions)
        
        return {
            'accuracy': accuracy * 100,
            'predictions': len(predictions),
            'classification_report': report,
            'confusion_matrix': cm,
            'model_type': self.model_type
        }
    
    def _save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
    
    def load_model(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            return True
        return False
    
    def get_feature_importance(self):
        """Get feature importance (for tree-based models)"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None

class ModelEnsemble:
    """Ensemble of multiple models for better performance"""
    
    def __init__(self, models=None):
        self.models = models or []
        self.is_trained = False
    
    def add_model(self, model):
        """Add a model to the ensemble"""
        self.models.append(model)
    
    def train_ensemble(self, X, y):
        """Train all models in the ensemble"""
        for model in self.models:
            model.train(X, y)
        self.is_trained = True
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise Exception("Ensemble must be trained before making predictions")
        
        predictions = []
        probabilities = []
        
        for model in self.models:
            pred, prob = model.predict(X)
            predictions.append(pred)
            probabilities.append(prob)
        
        # Average predictions (majority vote for classification)
        ensemble_pred = np.array(predictions).T
        ensemble_pred = [np.bincount(pred.astype(int)).argmax() for pred in ensemble_pred]
        
        # Average probabilities
        ensemble_prob = np.mean(probabilities, axis=0)
        
        return np.array(ensemble_pred), ensemble_prob
    
    def evaluate_ensemble(self, X, y):
        """Evaluate ensemble performance"""
        predictions, probabilities = self.predict_ensemble(X)
        
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        cm = confusion_matrix(y, predictions)
        
        return {
            'accuracy': accuracy * 100,
            'predictions': len(predictions),
            'classification_report': report,
            'confusion_matrix': cm,
            'model_type': 'ensemble'
        }

class ModelManager:
    """Manages multiple models and their configurations"""
    
    def __init__(self):
        self.models = {}
        self.active_model = None
    
    def create_model(self, name, model_type):
        """Create a new model"""
        model = BehaviorClassifier(model_type)
        self.models[name] = model
        return model
    
    def set_active_model(self, name):
        """Set the active model"""
        if name in self.models:
            self.active_model = self.models[name]
            return True
        return False
    
    def get_model_info(self, name):
        """Get information about a specific model"""
        if name in self.models:
            model = self.models[name]
            return {
                'name': name,
                'type': model.model_type,
                'is_trained': model.is_trained,
                'classes': model.classes,
                'model_path': model.model_path
            }
        return None
    
    def list_models(self):
        """List all available models"""
        return list(self.models.keys())
    
    def delete_model(self, name):
        """Delete a model"""
        if name in self.models:
            del self.models[name]
            if self.active_model == self.models.get(name):
                self.active_model = None
            return True
        return False
