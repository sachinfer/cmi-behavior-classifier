#!/usr/bin/env python3
"""
Test script to verify dashboard setup
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'torch', 
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    print("Testing package imports...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    return True

def test_model_loading():
    """Test if the LSTM model can be loaded"""
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        # Define the model class
        class LSTMClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n[-1])
        
        # Load the model
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['walking', 'sitting', 'driving'])
        
        model = LSTMClassifier(input_size=332, hidden_size=128, num_classes=len(label_encoder.classes_))
        model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
        model.eval()
        
        print("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Behavior Classification Dashboard Setup")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    print()
    
    # Test model loading
    model_ok = test_model_loading()
    print()
    
    # Summary
    if imports_ok and model_ok:
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\nTo run the dashboard:")
        print("  streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not imports_ok:
            print("\nTo install missing packages:")
            print("  pip install -r requirements.txt")
        if not model_ok:
            print("\nMake sure lstm_model.pth is in the current directory.")

if __name__ == "__main__":
    main() 