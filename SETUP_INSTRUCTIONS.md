# 🚀 Behavior Classification Dashboard - Setup Instructions

## 📁 Current Project Structure

Your project folder now contains:
```
cmi-behavior-classifier/
│
├── app.py                # ✅ Full LSTM dashboard (requires PyTorch)
├── app_simple.py         # ✅ Demo dashboard (works without PyTorch)
├── lstm_model.pth        # ✅ Your trained PyTorch model
├── requirements.txt      # ✅ Python dependencies
├── README.md            # ✅ Comprehensive documentation
├── test_setup.py        # ✅ Setup verification script
└── SETUP_INSTRUCTIONS.md # ✅ This file
```

## 🎉 Your Behavior Classification Dashboard is Ready!

I've successfully created a comprehensive behavior classification dashboard with all the features you requested. Here's what you now have:

### ✅ **Complete Project Structure:**
- ✅ **`app.py`** - Full LSTM dashboard with PyTorch integration
- ✅ **`app_simple.py`** - Demo version (works without PyTorch)
- ✅ **`lstm_model.pth`** - Your trained model (already present)
- ✅ **`requirements.txt`** - All dependencies listed
- ✅ **`README.md`** - Comprehensive documentation
- ✅ **`test_setup.py`** - Setup verification script
- ✅ **`SETUP_INSTRUCTIONS.md`** - Step-by-step setup guide

### 🚀 **Quick Start Options:**

**Option 1: Demo Version (Immediate)**
```bash
<code_block_to_apply_from>
streamlit run app_simple.py
```
This works right now without installing PyTorch!

**Option 2: Full Version (With LSTM Model)**
```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run full dashboard
streamlit run app.py
```

### ⚙️ **All Features Included:**
✅ Load your trained model  
✅ Upload sensor sequence CSV  
✅ Predict behavior with confidence scores  
✅ Live plotting of sensor data  
✅ Dataset exploration (EDA)  
✅ Sample testing from train.csv  
✅ Confusion matrix + accuracy  
✅ Download prediction results  
✅ Dark/light theme toggle  

### 📊 Supported Data Format:
```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,...
0.123,0.456,0.789,0.012,0.345,0.678,...
0.234,0.567,0.890,0.123,0.456,0.789,...
...
```

## 🛠️ Troubleshooting

### Common Issues:

**1. PyTorch Installation Fails**
```bash
# Try CPU version instead
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**2. File Permission Errors**
- Close any applications using Python
- Run as administrator if needed
- Try installing in a virtual environment

**3. Model Loading Errors**
- Ensure `lstm_model.pth` is in the same directory
- Check model architecture matches the code
- Verify PyTorch version compatibility

**4. Missing Dependencies**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### Virtual Environment Setup (Recommended):
```bash
# Create virtual environment
python -m venv behavior_env

# Activate (Windows)
behavior_env\Scripts\activate

# Activate (Linux/Mac)
source behavior_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run dashboard
streamlit run app.py
```

## 🎯 Usage Guide

### 1. Upload Data
- Use the file uploader to upload sensor CSV files
- View data preview and statistics

### 2. Explore Data
- See sensor data visualizations
- Check correlation matrices
- Analyze data distributions

### 3. Get Predictions
- Upload a sequence file
- View behavior predictions with confidence scores
- Download results as CSV

### 4. Test Model
- Load training data samples
- Compare true vs predicted labels
- View model performance metrics

### 5. Evaluate Performance
- Generate confusion matrices
- View classification reports
- Analyze accuracy metrics

## 📱 Browser Access

Once running, the dashboard will be available at:
- **Local**: `http://localhost:8501`
- **Network**: `http://your-ip:8501` (for sharing)

## 🔄 Next Steps

1. **Start with demo**: `streamlit run app_simple.py` (works immediately)
2. **Install PyTorch** when ready for full functionality
3. **Upload your CSV files** to test the dashboard
4. **Customize** the code for your specific needs

The dashboard will open in your browser at `http://localhost:8501` and provide a beautiful, interactive interface for behavior classification!

Would you like me to help you run the demo version now, or do you have any questions about the setup?

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Run `python test_setup.py` to diagnose problems
3. Ensure all files are in the correct directory
4. Verify Python and package versions

---

**Happy Classifying! 🎯**

Your behavior classification dashboard is ready to use! 