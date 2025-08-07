---
title: SenseBehavior
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: gradio
pinned: false
license: other
---

# 🚀 SenseBehavior - Human Behavior Classification Dashboard

A modern Flask-based web application that uses LSTM neural networks to classify human behavior patterns from sensor data. Features a beautiful, responsive interface with real-time analytics and visualization capabilities.

## ✨ Current Features

### 🎨 **Beautiful User Interface**
- **Modern Design**: Clean, professional interface with gradient backgrounds
- **Dark/Light Theme Toggle**: Switch between themes with the moon/sun button
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Beautiful File Upload**: Animated gradient button with hover effects and shimmer
- **Drag & Drop Support**: Intuitive file upload with visual feedback

### 📁 **File Upload & Processing**
- **CSV File Support**: Upload sensor data in CSV format
- **Drag & Drop Interface**: Simply drag files onto the upload area
- **File Validation**: Automatic validation of file format and structure
- **Real-time Feedback**: Loading indicators and progress updates
- **File Management**: Clear file selection and easy file removal

### 🧠 **Behavior Classification**
- **LSTM Neural Network**: Advanced deep learning model for pattern recognition
- **Four Behavior Types**: Walking, Sitting, Driving, Standing
- **Confidence Scores**: Detailed probability scores for each behavior
- **Real-time Prediction**: Instant results with confidence analysis
- **Demo Mode**: Works without PyTorch for testing purposes

### 📊 **Data Analytics & Visualization**
- **Data Overview**: Shape, columns, and basic statistics
- **Interactive Plots**: Matplotlib-generated visualizations
- **Data Preview**: Table view of uploaded data
- **Sensor Analysis**: Multi-axis sensor data visualization
- **Statistical Insights**: Comprehensive data analysis

### 🔍 **Model Evaluation**
- **Confusion Matrix**: Visual representation of model performance
- **Classification Report**: Precision, Recall, F1-Score metrics
- **Overall Accuracy**: Model performance summary
- **Sample Testing**: Test model on random training samples
- **Performance Analytics**: Detailed evaluation metrics

### ⚙️ **Configuration & Settings**
- **Model Parameters**: Adjustable input size, hidden size, and classes
- **Real-time Updates**: Configuration changes applied immediately
- **System Information**: PyTorch availability and training data status
- **Model Architecture**: View and modify LSTM model settings

### 📤 **Export & Download**
- **Prediction Results**: Download CSV files with prediction data
- **Confidence Analysis**: Export confidence threshold analysis
- **Model Metrics**: Download evaluation results and reports
- **Data Export**: Comprehensive data export capabilities

## 🎯 Supported Behaviors

The application can classify these human behaviors with high accuracy:

- **🚶‍♂️ Walking** - Movement patterns while walking
- **🪑 Sitting** - Stationary sitting behavior  
- **🚗 Driving** - Vehicle operation patterns
- **🧍‍♂️ Standing** - Upright stationary behavior

## 🚀 How to Use

### 1. **Access the Dashboard**
- Visit the application URL
- The interface loads with a beautiful gradient design

### 2. **Upload Your Data**
- **Option A**: Click the beautiful "Choose CSV File" button
- **Option B**: Drag and drop a CSV file onto the upload area
- Supported format: CSV files with sensor data

### 3. **View Results**
- **Data Overview**: See your data shape and column count
- **Visualizations**: Interactive plots of your sensor data
- **Predictions**: Behavior classification with confidence scores
- **Model Info**: Details about the LSTM model used

### 4. **Explore Features**
- **Model Evaluation**: Test the model on training data
- **Configuration**: Adjust model parameters
- **Export Results**: Download predictions and reports

## 📊 Data Format Requirements

Your CSV file should contain:
- **Numeric columns**: Sensor readings (accelerometer, gyroscope, etc.)
- **Time series data**: Sequential sensor measurements
- **332 features**: The model expects 332 input features
- **Optional columns**: sequence_id and behavior for evaluation

## 🛠️ Technical Architecture

- **Frontend**: HTML5, CSS3, JavaScript with Bootstrap 5
- **Backend**: Flask web framework with RESTful API
- **ML Model**: LSTM (Long Short-Term Memory) neural network
- **Visualization**: Matplotlib and Seaborn for interactive plots
- **Data Processing**: Pandas and NumPy for data manipulation
- **Evaluation**: Scikit-learn for metrics and analysis

## 🔧 API Endpoints

- `GET /` - Main dashboard interface
- `POST /upload` - Upload and process CSV files
- `GET /model_info` - Get model information
- `GET /evaluate` - Evaluate model performance
- `POST /test_sample` - Test model on random sample
- `POST /confidence_analysis` - Analyze confidence thresholds
- `POST /download_results` - Download prediction results
- `GET/POST /config` - Get/update model configuration
- `GET /health` - Health check endpoint

## 🎓 Academic Project

This project was developed for **CIS6005 — Computational Intelligence Project (2025)**.

## 📋 System Requirements

- **Python**: 3.11 or higher
- **Flask**: 2.3.3
- **PyTorch**: 2.0.1 (optional - demo mode available)
- **Pandas**: 2.0.3
- **Scikit-learn**: 1.3.0
- **NumPy**: 1.24.3
- **Matplotlib**: 3.7.2
- **Seaborn**: 0.12.2

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd SenseBehavior

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at http://localhost:7860
```

### Alternative: Run with Python directly
```bash
# Make sure you have Python 3.11+ installed
python --version

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

# The application will be available at http://localhost:7860
```

## 🎨 Interface Features

### **Upload & Predict Tab**
- Beautiful file upload interface
- Real-time data processing
- Interactive visualizations
- Behavior predictions with confidence scores

### **Model Evaluation Tab**
- Confusion matrix visualization
- Classification report with metrics
- Overall accuracy display
- Sample testing functionality

### **Configuration Tab**
- Model parameter adjustment
- System information display
- Real-time configuration updates
- Training data status

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Developed with ❤️ for Human Behavior Analysis**

*Built for CIS6005 — Computational Intelligence Project (2025)*
