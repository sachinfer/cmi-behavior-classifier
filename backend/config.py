"""
Configuration file for Behavior Analytics Dashboard backend
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'input_size': 332,
    'hidden_size': 128,
    'num_classes': 4,
    'classes': ['walking', 'sitting', 'driving', 'standing'],
    'sequence_length': 10,
    'dropout_rate': 0.2
}

# Data processing configuration
DATA_CONFIG = {
    'max_features': 332,
    'min_samples': 10,
    'validation_split': 0.2,
    'test_split': 0.2,
    'random_state': 42
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'max_features_heatmap': 20,
    'max_features_distribution': 6,
    'chart_height': 400,
    'chart_width': 800,
    'dpi': 100
}

# Model types and their configurations
MODEL_TYPES = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'neural_network': {
        'hidden_layer_sizes': (128, 64, 32),
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True
    },
    'svm': {
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'statistical_features': ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis'],
    'frequency_features': ['dominant_frequency', 'spectral_energy'],
    'time_features': ['autocorrelation', 'trend', 'seasonality']
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'app.log'
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 70.0,
    'min_precision': 0.7,
    'min_recall': 0.7,
    'min_f1_score': 0.7
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'max_missing_percentage': 5.0,
    'max_duplicate_percentage': 1.0,
    'min_numeric_features': 100,
    'min_samples_per_class': 10
}

# Export configuration
EXPORT_CONFIG = {
    'formats': ['CSV', 'JSON', 'Excel'],
    'include_timestamps': True,
    'include_confidence': True,
    'include_metadata': True
}

# API configuration (if needed in future)
API_CONFIG = {
    'host': 'localhost',
    'port': 8501,
    'debug': False,
    'reload': True
}
