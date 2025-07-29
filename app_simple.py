import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import io
import base64

# ----- HELPER FUNCTIONS -----
def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ----- SIDEBAR -----
st.sidebar.title("Sensor Behavior Classifier")
st.sidebar.info("Upload a `.csv` file or select a test sample.")

# ----- MAIN UI -----
st.title("🚀 Human Behavior Classification Dashboard")
st.write("Predict behavior using LSTM on sensor time-series data")

# Add a note about PyTorch installation
st.warning("""
⚠️ **Note**: This is a demonstration version. To use the full LSTM model functionality, 
please install PyTorch by running:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Then use the main `app.py` file instead.
""")

# ----- DATASET EXPLORATION -----
st.header("📊 Exploratory Data Analysis")
uploaded_data = st.file_uploader("Upload a sequence CSV file (one sample)", type=['csv'])

if uploaded_data:
    df = pd.read_csv(uploaded_data)
    st.write("### Uploaded Data Preview", df.head())
    st.write("Shape:", df.shape)

    # Plot selected sensor columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    st.write("### Sensor Data Over Time")
    
    # Create a multi-plot figure for better visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols[:6]):  # limit to first 6 for simplicity
        axes[i].plot(df[col])
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(numeric_cols[:6]), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Demo prediction (without actual model)
    st.write("### Demo Prediction (Mock)")
    st.info("This is a demonstration. Install PyTorch to use the real LSTM model.")
    
    # Mock prediction probabilities
    mock_probs = np.array([0.6, 0.3, 0.1])  # walking, sitting, driving
    mock_label = "walking"
    
    st.success(f"✅ Predicted Behavior: **{mock_label}**")
    st.write("### Prediction Probabilities:")
    
    # Create a more detailed probability visualization
    prob_df = pd.DataFrame({
        'Behavior': ['walking', 'sitting', 'driving'],
        'Probability': mock_probs
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(prob_df['Behavior'], prob_df['Probability'], 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidence Scores (Demo)')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, mock_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Download prediction results
    st.write("### Download Prediction Results")
    results_df = pd.DataFrame({
        'Predicted_Behavior': [mock_label],
        'Confidence': [max(mock_probs)],
        'All_Probabilities': [str(mock_probs.tolist())]
    })
    st.markdown(get_download_link(results_df, 'prediction_results.csv', 'Download Prediction Results'), unsafe_allow_html=True)

# ----- DEMO ON TRAIN DATA -----
st.header("🧪 Try a Sample From Dataset")
if st.checkbox("Load and test on sample from train.csv"):
    try:
        train_df = pd.read_csv("train.csv")
        st.write("### Training Data Preview")
        st.write(train_df.head())
        st.write("Shape:", train_df.shape)
        
        # Show some basic statistics
        st.write("### Data Statistics")
        st.write(train_df.describe())
        
        # Show behavior distribution
        if 'behavior' in train_df.columns:
            st.write("### Behavior Distribution")
            behavior_counts = train_df['behavior'].value_counts()
            st.bar_chart(behavior_counts)
            
    except FileNotFoundError:
        st.warning("⚠️ train.csv file not found. Please ensure it's in the same directory as app.py")

# ----- LIVE SENSOR DATA PLOTTING -----
st.header("📈 Live Sensor Data Visualization")
st.write("Upload a CSV file to see live plotting of sensor data")

if uploaded_data:
    df = pd.read_csv(uploaded_data)
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Create an interactive plot
    selected_sensors = st.multiselect(
        "Select sensors to plot:",
        numeric_cols.tolist(),
        default=numeric_cols[:3].tolist()
    )
    
    if selected_sensors:
        fig, ax = plt.subplots(figsize=(12, 6))
        for sensor in selected_sensors:
            ax.plot(df[sensor], label=sensor, alpha=0.8)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Sensor Values')
        ax.set_title('Live Sensor Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ----- DATA ANALYSIS TOOLS -----
st.header("🔍 Data Analysis Tools")

if uploaded_data:
    df = pd.read_csv(uploaded_data)
    
    # Correlation matrix
    st.write("### Correlation Matrix")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Sensor Correlation Matrix')
        st.pyplot(fig)
    
    # Statistical summary
    st.write("### Statistical Summary")
    st.write(df.describe())
    
    # Missing values
    st.write("### Missing Values")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.write(missing_data[missing_data > 0])
    else:
        st.success("✅ No missing values found!")

# ----- FOOTER -----
st.markdown("---")
st.caption("Developed for CIS6005 — Computational Intelligence Project (2025)")

# ----- ADDITIONAL INFO -----
with st.expander("ℹ️ About This Dashboard"):
    st.write("""
    This dashboard provides a comprehensive interface for human behavior classification using LSTM neural networks.
    
    **Features:**
    - ✅ Load your trained model
    - ✅ Upload sensor sequence CSV
    - ✅ Predict behavior with confidence scores
    - ✅ Live plotting of sensor data
    - ✅ Dataset exploration (EDA)
    - ✅ Sample testing from train.csv
    - ✅ Confusion matrix + accuracy
    - ✅ Download prediction results
    - ✅ Dark/light theme toggle (Streamlit built-in)
    
    **Usage:**
    1. Upload a CSV file with sensor data
    2. View the data visualization
    3. Get behavior predictions with confidence scores
    4. Test on samples from your training dataset
    5. Evaluate model performance with confusion matrix
    
    **To use the full LSTM model:**
    1. Install PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
    2. Run the main app: `streamlit run app.py`
    """) 