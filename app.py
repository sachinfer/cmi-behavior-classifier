from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Disable caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Global variables
model = None
label_encoder = None
scaler = None
pytorch_available = False
train_data = None
model_config = {
    'input_size': 332,
    'hidden_size': 128,
    'num_classes': 4,
    'classes': ['walking', 'sitting', 'driving', 'standing']
}

def load_model():
    global model, label_encoder, scaler, pytorch_available, train_data
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        pytorch_available = True
        
        print("PyTorch imported successfully")
        
        # Define LSTM model class with full capabilities
        class LSTMClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, num_classes)
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use the last hidden state
                last_hidden = h_n[-1]
                # Apply dropout
                dropped = self.dropout(last_hidden)
                # Final classification layer
                output = self.fc(dropped)
                return output, self.softmax(output)
        
        # Initialize components
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(model_config['classes'])
        
        print("Label encoder initialized")
        
        # Load model with full weights
        model = LSTMClassifier(
            input_size=model_config['input_size'], 
            hidden_size=model_config['hidden_size'], 
            num_classes=model_config['num_classes']
        )
        
        # Load the full model weights
        checkpoint = torch.load("lstm_model.pth", map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        model.eval()
        
        print("✅ Full model weights loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total parameters")
        
        scaler = StandardScaler()
        
        # Try to load training data if available
        try:
            if os.path.exists("train.csv"):
                train_data = pd.read_csv("train.csv")
                print("Training data loaded successfully")
        except Exception as e:
            print(f"Could not load training data: {e}")
        
        print("✅ Full model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        pytorch_available = False
        return False

def create_plot(data, title="Sensor Data", plot_type="line"):
    """Create matplotlib plots and return as base64 string"""
    plt.figure(figsize=(10, 6))
    
    if plot_type == "line":
        plt.plot(data)
    elif plot_type == "scatter":
        plt.scatter(range(len(data)), data, alpha=0.6)
    elif plot_type == "histogram":
        plt.hist(data, bins=30, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time Steps' if plot_type != "histogram" else 'Values')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3)
    
    # Save plot to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close()
    
    # Convert to base64
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def predict_behavior(data, confidence_threshold=0.7):
    """Predict behavior from sensor data with full model weights"""
    if not pytorch_available or model is None:
        return {
            'prediction': 'walking',
            'confidence': 0.85,
            'probabilities': {'walking': 0.85, 'sitting': 0.10, 'driving': 0.03, 'standing': 0.02},
            'mode': 'demo',
            'model_used': 'demo'
        }
    
    try:
        import torch
        
        # Preprocess data with full feature engineering
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Remove non-sensor columns if they exist
        sensor_cols = [col for col in numeric_cols if col not in ['sequence_id', 'behavior']]
        
        if len(sensor_cols) == 0:
            sensor_cols = numeric_cols
        
        # Prepare input data
        input_data = data[sensor_cols].values
        
        # Handle different input sizes
        if input_data.shape[1] != model_config['input_size']:
            if input_data.shape[1] > model_config['input_size']:
                # Truncate if too many features
                input_data = input_data[:, :model_config['input_size']]
            else:
                # Pad with zeros if too few features
                padded = np.zeros((input_data.shape[0], model_config['input_size']))
                padded[:, :input_data.shape[1]] = input_data
                input_data = padded
        
        # Standardize the data
        input_scaled = scaler.fit_transform(input_data)
        
        # Convert to tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction with full model
        with torch.no_grad():
            raw_output, probabilities = model(input_tensor)
            probs = probabilities.numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(max(probs))
        
        # Create probabilities dict
        prob_dict = {label: float(prob) for label, prob in zip(label_encoder.classes_, probs)}
        
        # Determine prediction quality
        if confidence >= confidence_threshold:
            prediction_quality = "high_confidence"
        elif confidence >= 0.5:
            prediction_quality = "medium_confidence"
        else:
            prediction_quality = "low_confidence"
        
        return {
            'prediction': pred_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'mode': 'real',
            'model_used': 'full_lstm',
            'prediction_quality': prediction_quality,
            'input_features': len(sensor_cols),
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            'prediction': 'walking',
            'confidence': 0.85,
            'probabilities': {'walking': 0.85, 'sitting': 0.10, 'driving': 0.03, 'standing': 0.02},
            'mode': 'error',
            'model_used': 'fallback',
            'error': str(e)
        }

def evaluate_model():
    """Evaluate model on training data"""
    if train_data is None or not pytorch_available:
        return None
    
    try:
        import torch
        
        # Prepare training data
        sensor_cols = train_data.columns[train_data.columns.str.contains('acc|gyro|mag')]
        if len(sensor_cols) == 0:
            sensor_cols = train_data.select_dtypes(include=np.number).columns
        
        # Group by sequence and get labels
        sequences = []
        labels = []
        
        if 'sequence_id' in train_data.columns and 'behavior' in train_data.columns:
            for seq_id in train_data['sequence_id'].unique():
                seq_data = train_data[train_data['sequence_id'] == seq_id]
                if len(seq_data) > 0:
                    sequences.append(seq_data[sensor_cols].values)
                    labels.append(seq_data['behavior'].iloc[0])
        else:
            # Fallback: treat each row as a sequence
            sequences = [train_data[sensor_cols].values]
            labels = ['unknown']
        
        # Encode labels
        label_encoder.fit(labels)
        labels_encoded = label_encoder.transform(labels)
        
        # Make predictions
        predictions = []
        true_labels = []
        
        for i, seq in enumerate(sequences):
            try:
                # Pad sequence if needed
                if seq.shape[0] < model_config['input_size']:
                    # Pad with zeros
                    padded = np.zeros((model_config['input_size'], seq.shape[1]))
                    padded[:seq.shape[0], :] = seq
                else:
                    # Truncate if too long
                    padded = seq[:model_config['input_size'], :]
                
                # Standardize
                padded_scaled = scaler.fit_transform(padded)
                input_tensor = torch.tensor(padded_scaled, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    predictions.append(pred)
                    true_labels.append(labels_encoded[i])
            except Exception as e:
                print(f"Error predicting sequence {i}: {e}")
                continue
        
        if len(predictions) > 0:
            # Calculate metrics
            cm = confusion_matrix(true_labels, predictions)
            report = classification_report(true_labels, predictions, target_names=label_encoder.classes_, output_dict=True)
            
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=label_encoder.classes_, 
                       yticklabels=label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            cm_plot = io.BytesIO()
            plt.savefig(cm_plot, format='png', bbox_inches='tight')
            cm_plot.seek(0)
            plt.close()
            
            cm_base64 = base64.b64encode(cm_plot.getvalue()).decode()
            
            return {
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'accuracy': (cm.diagonal().sum() / cm.sum()) * 100,
                'cm_plot': cm_base64,
                'predictions': len(predictions)
            }
        
    except Exception as e:
        print(f"Error in model evaluation: {e}")
    
    return None

@app.route('/')
def index():
    print(f"Index route accessed at {datetime.now()}")
    return render_template('index.html', 
                         pytorch_available=pytorch_available,
                         model_config=model_config,
                         train_data_available=train_data is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f"Upload route accessed at {datetime.now()}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Basic data info
        data_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=np.number).columns.tolist(),
            'preview': df.head().to_html(classes='table table-striped')
        }
        
        # Create visualizations
        plots = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 0:
            # Plot first few numeric columns
            for i, col in enumerate(numeric_cols[:4]):
                plots[f'plot_{i}'] = {
                    'data': create_plot(df[col].values, f'{col} Over Time'),
                    'title': f'{col} Over Time',
                    'column': col
                }
        
        # Make prediction
        prediction = predict_behavior(df)
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'plots': plots,
            'prediction': prediction
        })
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/evaluate')
def evaluate():
    """Evaluate model on training data"""
    print(f"Evaluate route accessed at {datetime.now()}")
    evaluation = evaluate_model()
    
    if evaluation:
        return jsonify({
            'success': True,
            'evaluation': evaluation
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Could not evaluate model. Training data may not be available.'
        })

@app.route('/test_sample')
def test_sample():
    """Test model on a sample from training data"""
    print(f"Test sample route accessed at {datetime.now()}")
    
    if train_data is None:
        return jsonify({'error': 'Training data not available'}), 400
    
    try:
        # Get a random sample
        sample = train_data.sample(n=1)
        numeric_cols = sample.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 0:
            # Create a small dataset for prediction
            sample_data = sample[numeric_cols]
            prediction = predict_behavior(sample_data)
            
            return jsonify({
                'success': True,
                'sample_shape': sample.shape,
                'prediction': prediction
            })
        else:
            return jsonify({'error': 'No numeric columns found in sample'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error testing sample: {str(e)}'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download prediction results as CSV"""
    try:
        data = request.json
        if not data or 'results' not in data:
            return jsonify({'error': 'No results data provided'}), 400
        
        # Create DataFrame from results
        df = pd.DataFrame(data['results'])
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='prediction_results.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Error creating download: {str(e)}'}), 500

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update model configuration"""
    global model_config
    
    if request.method == 'POST':
        try:
            new_config = request.json
            model_config.update(new_config)
            return jsonify({'success': True, 'config': model_config})
        except Exception as e:
            return jsonify({'error': f'Error updating config: {str(e)}'}), 500
    
    return jsonify({'config': model_config})

@app.route('/health')
def health():
    print(f"Health check at {datetime.now()}")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pytorch_available': pytorch_available,
        'model_loaded': model is not None,
        'train_data_available': train_data is not None
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'Flask app is running!',
        'timestamp': datetime.now().isoformat(),
        'pytorch_available': pytorch_available,
        'model_loaded': model is not None
    })

@app.route('/model_info')
def model_info():
    """Get detailed information about the loaded model"""
    if not pytorch_available or model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        })
    
    try:
        import torch
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model architecture info
        model_info = {
            'model_type': 'LSTM Classifier',
            'input_size': model_config['input_size'],
            'hidden_size': model_config['hidden_size'],
            'num_classes': model_config['num_classes'],
            'classes': model_config['classes'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': {
                'lstm_layers': 1,
                'dropout': 0.2,
                'activation': 'Softmax'
            },
            'status': 'loaded',
            'pytorch_version': torch.__version__
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict behavior for multiple sequences at once"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check if we have sequence_id column
        if 'sequence_id' not in df.columns:
            return jsonify({'error': 'CSV must contain sequence_id column for batch processing'}), 400
        
        # Group by sequence_id
        sequences = []
        results = []
        
        for seq_id in df['sequence_id'].unique():
            seq_data = df[df['sequence_id'] == seq_id]
            if len(seq_data) > 0:
                # Make prediction for this sequence
                prediction = predict_behavior(seq_data)
                
                results.append({
                    'sequence_id': int(seq_id),
                    'data_points': len(seq_data),
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'probabilities': prediction['probabilities'],
                    'prediction_quality': prediction.get('prediction_quality', 'unknown'),
                    'model_used': prediction.get('model_used', 'unknown')
                })
        
        return jsonify({
            'success': True,
            'total_sequences': len(results),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing batch: {str(e)}'}), 500

@app.route('/confidence_analysis', methods=['POST'])
def confidence_analysis():
    """Analyze prediction confidence across different thresholds"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Test different confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        analysis = []
        
        for threshold in thresholds:
            prediction = predict_behavior(df, confidence_threshold=threshold)
            
            analysis.append({
                'threshold': threshold,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'quality': prediction.get('prediction_quality', 'unknown'),
                'above_threshold': prediction['confidence'] >= threshold
            })
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'recommended_threshold': 0.7
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in confidence analysis: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting SenseBehavior Flask Application...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    load_model()
    
    port = int(os.environ.get('PORT', 7860))
    print(f"Starting Flask server on port {port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        sys.exit(1) 