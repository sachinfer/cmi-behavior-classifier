"""
Data processing module for sensor data analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_data(self, data):
        """Validate data structure and content"""
        if data.empty:
            raise ValueError("Data is empty")
        
        # Check for required columns
        required_features = 332  # Expected number of features
        
        if len(data.columns) < required_features:
            raise ValueError(f"Data must have at least {required_features} features")
        
        # Check for numeric data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < required_features:
            raise ValueError(f"At least {required_features} numeric columns required")
        
        return True
    
    def preprocess_data(self, data, target_column=None):
        """Preprocess data for model input"""
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Ensure we have the right number of features
        if len(numeric_data.columns) > 332:
            # Select first 332 features
            numeric_data = numeric_data.iloc[:, :332]
        elif len(numeric_data.columns) < 332:
            # Pad with zeros if fewer features
            padding = pd.DataFrame(
                np.zeros((len(numeric_data), 332 - len(numeric_data.columns))),
                columns=[f'feature_{i}' for i in range(len(numeric_data.columns), 332)]
            )
            numeric_data = pd.concat([numeric_data, padding], axis=1)
        
        # Scale features
        if not self.is_fitted:
            scaled_data = self.scaler.fit_transform(numeric_data)
            self.is_fitted = True
        else:
            scaled_data = self.scaler.transform(numeric_data)
        
        return scaled_data
    
    def prepare_sequences(self, data, sequence_length=10):
        """Prepare sequential data for LSTM model"""
        sequences = []
        
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def extract_features(self, data):
        """Extract statistical features from sensor data"""
        features = {}
        
        # Basic statistics
        features['mean'] = data.mean()
        features['std'] = data.std()
        features['min'] = data.min()
        features['max'] = data.max()
        features['median'] = data.median()
        
        # Additional features
        features['skewness'] = data.skew()
        features['kurtosis'] = data.kurtosis()
        features['range'] = data.max() - data.min()
        features['iqr'] = data.quantile(0.75) - data.quantile(0.25)
        
        return features
    
    def analyze_sensor_patterns(self, data):
        """Analyze patterns in sensor data"""
        analysis = {}
        
        # Frequency domain analysis (if applicable)
        if len(data) > 1:
            # Calculate FFT for time series analysis
            fft_values = np.fft.fft(data.iloc[:, 0])  # First column
            analysis['dominant_frequency'] = np.argmax(np.abs(fft_values[1:len(fft_values)//2])) + 1
            analysis['spectral_energy'] = np.sum(np.abs(fft_values)**2)
        
        # Statistical patterns
        analysis['autocorrelation'] = data.iloc[:, 0].autocorr() if len(data) > 1 else 0
        
        return analysis
    
    def create_sample_data(self, num_samples=100, num_features=332):
        """Create sample sensor data for testing"""
        np.random.seed(42)
        
        # Generate realistic sensor data
        data = np.random.randn(num_samples, num_features)
        
        # Add some patterns to make it more realistic
        for i in range(num_features):
            if i % 3 == 0:  # Every 3rd feature has some trend
                data[:, i] += np.linspace(0, 2, num_samples)
            elif i % 5 == 0:  # Every 5th feature has some periodicity
                data[:, i] += 0.5 * np.sin(np.linspace(0, 4*np.pi, num_samples))
        
        # Create column names
        columns = [f'feature_{i}' for i in range(num_features)]
        
        return pd.DataFrame(data, columns=columns)

class DataValidator:
    """Validates data quality and structure"""
    
    @staticmethod
    def check_data_quality(data):
        """Check overall data quality"""
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Calculate missing percentage
        quality_report['missing_percentage'] = (quality_report['missing_values'] / 
                                              (quality_report['total_rows'] * quality_report['total_columns'])) * 100
        
        return quality_report
    
    @staticmethod
    def suggest_improvements(quality_report):
        """Suggest data quality improvements"""
        suggestions = []
        
        if quality_report['missing_percentage'] > 5:
            suggestions.append("High percentage of missing values. Consider imputation strategies.")
        
        if quality_report['duplicate_rows'] > 0:
            suggestions.append("Duplicate rows detected. Consider removing duplicates.")
        
        if quality_report['numeric_columns'] < 100:
            suggestions.append("Low number of numeric features. Check if data is properly formatted.")
        
        return suggestions
