# 🚀 Human Behavior Classification Dashboard

A comprehensive Streamlit dashboard for predicting human behavior using LSTM neural networks on sensor time-series data.

## ⚙️ Features

✅ **Load your trained model** - Seamlessly load and use your pre-trained LSTM model  
✅ **Upload sensor sequence CSV** - Easy file upload for new predictions  
✅ **Predict behavior with confidence scores** - Get detailed probability distributions  
✅ **Live plotting of sensor data** - Interactive visualizations of sensor readings  
✅ **Dataset exploration (EDA)** - Comprehensive data analysis tools  
✅ **Sample testing from train.csv** - Test model on known samples  
✅ **Confusion matrix + accuracy** - Model evaluation and performance metrics  
✅ **Download prediction results** - Export predictions as CSV files  
✅ **Dark/light theme toggle** - Streamlit built-in theme switching  

## 📁 Project Structure

```
project-folder/
│
├── app.py                # Main Streamlit dashboard application
├── lstm_model.pth        # Your trained PyTorch model
├── train.csv             # Training data (for EDA/demo)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit torch pandas numpy scikit-learn matplotlib seaborn
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### Upload and Predict
1. **Upload CSV File**: Use the file uploader to upload a sensor sequence CSV
2. **View Data**: Explore the uploaded data with interactive visualizations
3. **Get Predictions**: See behavior predictions with confidence scores
4. **Download Results**: Export prediction results as CSV

### Test on Training Data
1. **Enable Testing**: Check the "Load and test on sample from train.csv" option
2. **Select Sample**: Use the slider to choose different samples
3. **Compare Results**: See true vs predicted labels with accuracy indicators

### Model Evaluation
1. **Show Confusion Matrix**: Enable to see model performance metrics
2. **View Accuracy**: See overall accuracy and per-class performance
3. **Classification Report**: Detailed precision, recall, and F1 scores

### Live Data Visualization
1. **Select Sensors**: Choose which sensor data to plot
2. **Interactive Plots**: View real-time sensor data visualizations
3. **Multi-sensor View**: Compare multiple sensors simultaneously

## 🔧 Configuration

### Model Parameters
The dashboard is configured for:
- **Input Size**: 332 features
- **Hidden Size**: 128 LSTM units
- **Classes**: walking, sitting, driving

To modify these parameters, edit the `LSTMClassifier` class in `app.py`:

```python
model = LSTMClassifier(input_size=332, hidden_size=128, num_classes=3)
```

### Label Classes
Update the label encoder classes in `app.py`:

```python
label_encoder.classes_ = np.array(['walking', 'sitting', 'driving'])
```

## 📈 Data Format

### Input CSV Format
Your CSV file should contain:
- **Numeric columns**: Sensor data (accelerometer, gyroscope, etc.)
- **Time series**: Each row represents a time step
- **Consistent features**: Same number of features as training data

### Example CSV Structure
```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,...
0.123,0.456,0.789,0.012,0.345,0.678,...
0.234,0.567,0.890,0.123,0.456,0.789,...
...
```

## 🛠️ Troubleshooting

### Common Issues

**Model Loading Error**
- Ensure `lstm_model.pth` is in the same directory as `app.py`
- Check that the model architecture matches the `LSTMClassifier` class

**Missing Dependencies**
- Run `pip install -r requirements.txt`
- Ensure all packages are compatible versions

**CSV Upload Issues**
- Verify CSV format matches expected structure
- Check for missing or non-numeric columns

**Memory Issues**
- Reduce batch size for large datasets
- Close other applications to free memory

## 📝 Development

### Adding New Features
1. **New Visualizations**: Add matplotlib/seaborn plots
2. **Additional Metrics**: Extend evaluation section
3. **Custom Preprocessing**: Modify data processing pipeline

### Customization
- **Styling**: Modify Streamlit components and CSS
- **Layout**: Reorganize sections and widgets
- **Functionality**: Add new prediction methods or data sources

## 📚 Dependencies

- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Plotting library
- **Seaborn**: Statistical data visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for CIS6005 — Computational Intelligence Project (2025)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue with detailed error information

---

**Happy Classifying! 🎯** 