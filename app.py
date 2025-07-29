import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import io
import base64

# ----- MODEL CLASS -----
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ----- LOAD LABEL ENCODER -----
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['walking', 'sitting', 'driving'])  # Replace with your actual class names

# ----- LOAD MODEL -----
@st.cache_resource
def load_model():
    model = LSTMClassifier(input_size=332, hidden_size=128, num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

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

    # Normalize + Predict
    scaler = StandardScaler()
    input_data = scaler.fit_transform(df[numeric_cols].values)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"✅ Predicted Behavior: **{pred_label}**")
    st.write("### Prediction Probabilities:")
    
    # Create a more detailed probability visualization
    prob_df = pd.DataFrame({
        'Behavior': label_encoder.classes_,
        'Probability': probs
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(prob_df['Behavior'], prob_df['Probability'], 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidence Scores')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Download prediction results
    st.write("### Download Prediction Results")
    results_df = pd.DataFrame({
        'Predicted_Behavior': [pred_label],
        'Confidence': [max(probs)],
        'All_Probabilities': [str(probs.tolist())]
    })
    st.markdown(get_download_link(results_df, 'prediction_results.csv', 'Download Prediction Results'), unsafe_allow_html=True)

# ----- DEMO ON TRAIN DATA -----
st.header("🧪 Try a Sample From Dataset")
if st.checkbox("Load and test on sample from train.csv"):
    try:
        train_df = pd.read_csv("train.csv")
        sensor_cols = train_df.columns[train_df.columns.get_loc('acc_x'):]
        sequences = []
        labels = []
        label_df = train_df[['sequence_id', 'behavior']].drop_duplicates()
        label_encoder.fit(label_df['behavior'])

        for seq_id in label_df['sequence_id']:
            seq_data = train_df[train_df['sequence_id'] == seq_id][sensor_cols].values
            sequences.append(seq_data)
            labels.append(label_df[label_df['sequence_id'] == seq_id]['behavior'].values[0])
        scaler.fit(np.vstack(sequences))

        idx = st.slider("Choose a sample index", 0, len(sequences) - 1, 0)
        sample = scaler.transform(sequences[idx])
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(sample_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            true_label = labels[idx]

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**True Label:** {true_label}")
            st.write(f"**Predicted:** {pred_label}")
            st.write(f"**Confidence:** {max(probs):.3f}")
            
            # Color code the prediction
            if pred_label == true_label:
                st.success("✅ Correct Prediction!")
            else:
                st.error("❌ Incorrect Prediction")
        
        with col2:
            st.write("### Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Behavior': label_encoder.classes_,
                'Probability': probs
            })
            st.bar_chart(prob_df.set_index('Behavior'))
            
    except FileNotFoundError:
        st.warning("⚠️ train.csv file not found. Please ensure it's in the same directory as app.py")

# ----- CONFUSION MATRIX -----
st.header("📉 Model Evaluation on Validation Set")
if st.checkbox("Show Confusion Matrix"):
    try:
        train_df = pd.read_csv("train.csv")
        sensor_cols = train_df.columns[train_df.columns.get_loc('acc_x'):]
        sequences = []
        labels = []
        label_df = train_df[['sequence_id', 'behavior']].drop_duplicates()
        label_encoder.fit(label_df['behavior'])

        for seq_id in label_df['sequence_id']:
            seq_data = train_df[train_df['sequence_id'] == seq_id][sensor_cols].values
            sequences.append(seq_data)
            labels.append(label_df[label_df['sequence_id'] == seq_id]['behavior'].values[0])

        labels_encoded = label_encoder.transform(labels)
        padded = pad_sequence([torch.tensor(s, dtype=torch.float32) for s in sequences], batch_first=True)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            padded, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
        )

        preds, targets = [], []
        with torch.no_grad():
            for i in range(len(X_val)):
                x = X_val[i].unsqueeze(0)
                output = model(x)
                pred = torch.argmax(output, dim=1).item()
                preds.append(pred)
                targets.append(y_val[i])

        cm = confusion_matrix(targets, preds)
        
        # Calculate accuracy
        accuracy = (cm.diagonal().sum() / cm.sum()) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Overall Accuracy: {accuracy:.2f}%**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=label_encoder.classes_, 
                       yticklabels=label_encoder.classes_)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        with col2:
            st.write("### Classification Report:")
            report = classification_report(targets, preds, target_names=label_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
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
    """) 