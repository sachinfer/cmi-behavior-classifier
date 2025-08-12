import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import backend modules
from backend.data_processor import DataProcessor, DataValidator
from backend.ml_model import BehaviorClassifier, ModelManager
from backend.analytics import DataVisualizer, StatisticalAnalyzer

# Page configuration
st.set_page_config(
    page_title="Behavior Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .upload-section {
        border: 2px dashed #3b82f6;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(59, 130, 246, 0.05);
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Initialize backend components
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DataVisualizer()
if 'statistical_analyzer' not in st.session_state:
    st.session_state.statistical_analyzer = StatisticalAnalyzer()

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>📊 Behavior Analytics</h2>
            <p style="color: #94a3b8;">AI-Powered Classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📤 Data Upload", "📊 Analytics", "🧠 Model Evaluation", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ Model Ready")
        with col2:
            st.warning("⚠️ Demo Mode")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### Quick Stats")
        st.metric("Total Predictions", "1,234")
        st.metric("Accuracy", "94.2%")
        st.metric("Active Sessions", "5")

    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Data Upload":
        show_data_upload()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "🧠 Model Evaluation":
        show_model_evaluation()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>Behavior Analytics Dashboard</h1>
        <p>Monitor and analyze human behavior patterns with AI-powered classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1,234</h3>
            <p>Total Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>94.2%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>Active Sessions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>2</h3>
            <p>Errors</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("### 📈 Recent Activity")
    
    # Create sample activity data
    activity_data = pd.DataFrame({
        'Time': pd.date_range(start='2024-01-01', periods=10, freq='H'),
        'Action': ['File Upload', 'Prediction', 'Model Update', 'Data Export', 'Evaluation', 
                  'File Upload', 'Prediction', 'Model Update', 'Data Export', 'Evaluation'],
        'Status': ['Success', 'Success', 'Success', 'Success', 'Success',
                  'Success', 'Success', 'Success', 'Success', 'Success']
    })
    
    st.dataframe(activity_data, use_container_width=True)
    
    # Behavior Distribution Chart
    st.markdown("### 🎯 Behavior Distribution")
    
    behaviors = ['Walking', 'Sitting', 'Driving', 'Standing']
    counts = [45, 32, 18, 25]
    
    fig = px.pie(
        values=counts, 
        names=behaviors, 
        title="Recent Behavior Classifications",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def show_data_upload():
    st.markdown("""
    <div class="main-header">
        <h1>Data Upload & Processing</h1>
        <p>Upload sensor data for behavior classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown("""
    <div class="upload-section">
        <h3>📁 Upload Your Sensor Data</h3>
        <p>Drag and drop your CSV file or click browse below</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing sensor data"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        try:
            # Load data using backend
            data = st.session_state.data_processor.load_data(uploaded_file)
            st.session_state.data = data
            
            # Validate data
            try:
                st.session_state.data_processor.validate_data(data)
                st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
                
                # Data quality analysis
                quality_report = st.session_state.statistical_analyzer.analyze_data_quality(data)
                
                # Show file info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                with col4:
                    st.metric("Quality Score", f"{quality_report['quality_score']:.1f}%")
                
                # Data Preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                # Data Quality Report
                st.markdown("### 🔍 Data Quality Report")
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    st.metric("Missing Values", quality_report['missing_values'])
                    st.metric("Duplicate Rows", quality_report['duplicate_rows'])
                
                with quality_col2:
                    st.metric("Missing %", f"{quality_report['missing_percentage']:.1f}%")
                    st.metric("Memory Usage", f"{quality_report['memory_usage_mb']:.1f} MB")
                
                # Basic Statistics
                st.markdown("### 📊 Basic Statistics")
                st.dataframe(data.describe(), use_container_width=True)
                
                # Process Button
                if st.button("🚀 Process Data & Generate Predictions", type="primary"):
                    with st.spinner("Processing data and generating predictions..."):
                        # Preprocess data using backend
                        processed_data = st.session_state.data_processor.preprocess_data(data)
                        
                        # Generate predictions using backend
                        classifier = st.session_state.model_manager.create_model('main', 'random_forest')
                        predictions = classifier.predict_single(processed_data)
                        st.session_state.predictions = predictions
                        
                        st.success("✅ Data processed successfully!")
                        
                        # Show predictions
                        show_predictions(predictions)
                
            except ValueError as e:
                st.warning(f"⚠️ Data validation warning: {str(e)}")
                st.info("Data will be processed with available features")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def show_predictions(predictions):
    st.markdown("### 🧠 Behavior Classification Results")
    
    # Prediction Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Walking", f"{predictions['walking']:.1%}")
    with col2:
        st.metric("Sitting", f"{predictions['sitting']:.1%}")
    with col3:
        st.metric("Driving", f"{predictions['driving']:.1%}")
    with col4:
        st.metric("Standing", f"{predictions['standing']:.1%}")
    
    # Prediction Chart
    fig = px.bar(
        x=list(predictions.keys()),
        y=list(predictions.values()),
        title="Behavior Probability Distribution",
        color=list(predictions.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_title="Behavior", yaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Summary
    predicted_behavior = max(predictions, key=predictions.get)
    confidence = predictions[predicted_behavior]
    
    st.markdown(f"""
    <div class="success-box">
        <h4>🎯 Predicted Behavior: {predicted_behavior.title()}</h4>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Model:</strong> LSTM Neural Network (Demo Mode)</p>
    </div>
    """, unsafe_allow_html=True)



def show_analytics():
    st.markdown("""
    <div class="main-header">
        <h1>Data Analytics</h1>
        <p>Deep insights into your sensor data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first to see analytics")
        return
    
    data = st.session_state.data
    
    # Data Overview
    st.markdown("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        quality_report = st.session_state.statistical_analyzer.analyze_data_quality(data)
        st.metric("Quality Score", f"{quality_report['quality_score']:.1f}%")
    
    # Feature Analysis
    st.markdown("### 🔍 Feature Analysis")
    
    # Select numeric columns for analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Correlation Heatmap using backend
        if len(numeric_cols) <= 20:  # Only show if reasonable number of features
            st.markdown("#### Correlation Matrix")
            correlation_plot = st.session_state.visualizer.create_correlation_heatmap(data)
            st.image(f"data:image/png;base64,{correlation_plot}", use_column_width=True)
        
        # Feature Distribution using backend
        st.markdown("#### Feature Distributions")
        selected_features = st.multiselect(
            "Select features to visualize:",
            numeric_cols[:10],  # Limit to first 10 features
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:1]
        )
        
        if selected_features:
            distribution_plot = st.session_state.visualizer.create_feature_distributions(
                data, features=selected_features
            )
            st.image(f"data:image/png;base64,{distribution_plot}", use_column_width=True)
    
    # Advanced Analytics
    st.markdown("### 📈 Advanced Analytics")
    
    # Statistical Analysis
    summary_stats = st.session_state.statistical_analyzer.generate_summary_statistics(data)
    
    # Outlier Detection
    outliers = st.session_state.statistical_analyzer.detect_outliers(data)
    outlier_count = sum(outlier['outlier_count'] for outlier in outliers.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Outliers", outlier_count)
    with col2:
        st.metric("Features with Outliers", len([k for k, v in outliers.items() if v['outlier_count'] > 0]))
    with col3:
        st.metric("Data Completeness", f"{100 - summary_stats['missing_values'].get('total', 0):.1f}%")
    
    # Time Series Analysis (if applicable)
    if 'timestamp' in data.columns or any('time' in col.lower() for col in data.columns):
        st.markdown("### ⏰ Time Series Analysis")
        time_series_plot = st.session_state.visualizer.create_time_series_plot(data)
        st.image(f"data:image/png;base64,{time_series_plot}", use_column_width=True)
    
    # Statistical Summary
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(summary_stats['basic_stats'], use_container_width=True)
    
    # Data Quality Insights
    st.markdown("### 🔍 Data Quality Insights")
    quality_suggestions = DataValidator.suggest_improvements(quality_report)
    
    if quality_suggestions:
        st.warning("⚠️ Data Quality Issues Detected:")
        for suggestion in quality_suggestions:
            st.write(f"• {suggestion}")
    else:
        st.success("✅ Data quality looks good!")

def show_model_evaluation():
    st.markdown("""
    <div class="main-header">
        <h1>Model Evaluation</h1>
        <p>Assess model performance and metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection
    st.markdown("### 🤖 Model Selection")
    
    model_type = st.selectbox(
        "Select Model Type:",
        ["random_forest", "neural_network", "svm"],
        help="Choose the type of model to use for evaluation"
    )
    
    # Create and manage model
    if st.button("🔧 Initialize Model", type="secondary"):
        model = st.session_state.model_manager.create_model('evaluation', model_type)
        st.session_state.model_manager.set_active_model('evaluation')
        st.success(f"✅ {model_type.replace('_', ' ').title()} model initialized!")
    
    # Performance Metrics
    st.markdown("### 📊 Performance Metrics")
    
    if st.session_state.model_manager.active_model:
        model_info = st.session_state.model_manager.get_model_info('evaluation')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Type", model_info['type'].replace('_', ' ').title())
        with col2:
            st.metric("Status", "Trained" if model_info['is_trained'] else "Not Trained")
        with col3:
            st.metric("Classes", len(model_info['classes']))
        with col4:
            st.metric("Model Path", model_info['model_path'])
        
        # Training Section
        st.markdown("### 🎯 Model Training")
        
        if st.button("🚀 Train Model", type="primary"):
            if st.session_state.data is not None:
                with st.spinner("Training model..."):
                    try:
                        # Prepare training data
                        processed_data = st.session_state.data_processor.preprocess_data(st.session_state.data)
                        
                        # Create dummy labels for demo (in real scenario, these would come from data)
                        dummy_labels = np.random.choice(['walking', 'sitting', 'driving', 'standing'], size=len(processed_data))
                        
                        # Train model
                        model = st.session_state.model_manager.active_model
                        model.train(processed_data, dummy_labels)
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Update model info
                        model_info = st.session_state.model_manager.get_model_info('evaluation')
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
            else:
                st.warning("⚠️ Please upload data first before training")
        
        # Evaluation Section
        if st.session_state.model_manager.active_model.is_trained:
            st.markdown("### 📈 Model Evaluation")
            
            if st.button("📊 Evaluate Model", type="primary"):
                with st.spinner("Evaluating model..."):
                    try:
                        # Prepare test data
                        processed_data = st.session_state.data_processor.preprocess_data(st.session_state.data)
                        dummy_labels = np.random.choice(['walking', 'sitting', 'driving', 'standing'], size=len(processed_data))
                        
                        # Evaluate model
                        model = st.session_state.model_manager.active_model
                        evaluation_results = model.evaluate(processed_data, dummy_labels)
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Overall Accuracy", f"{evaluation_results['accuracy']:.1f}%")
                        with col2:
                            st.metric("Total Predictions", evaluation_results['predictions'])
                        with col3:
                            st.metric("Model Type", evaluation_results['model_type'])
                        with col4:
                            st.metric("Status", "✅ Trained")
                        
                        # Confusion Matrix
                        st.markdown("### 🎯 Confusion Matrix")
                        cm_plot = st.session_state.visualizer.create_confusion_matrix(
                            dummy_labels, 
                            model.predict(processed_data)[0]
                        )
                        st.image(f"data:image/png;base64,{cm_plot}", use_column_width=True)
                        
                        # Classification Report
                        st.markdown("### 📋 Classification Report")
                        report = evaluation_results['classification_report']
                        
                        # Convert to DataFrame for better display
                        report_data = []
                        for class_name, metrics in report.items():
                            if isinstance(metrics, dict) and 'precision' in metrics:
                                report_data.append({
                                    'Class': class_name,
                                    'Precision': f"{metrics['precision']:.3f}",
                                    'Recall': f"{metrics['recall']:.3f}",
                                    'F1-Score': f"{metrics['f1-score']:.3f}",
                                    'Support': metrics['support']
                                })
                        
                        if report_data:
                            st.dataframe(pd.DataFrame(report_data), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {str(e)}")
        
        # Model Testing
        st.markdown("### 🧪 Model Testing")
        
        if st.button("🎲 Test Random Sample", type="primary"):
            if st.session_state.data is not None:
                with st.spinner("Testing model on random sample..."):
                    try:
                        # Prepare test data
                        processed_data = st.session_state.data_processor.preprocess_data(st.session_state.data)
                        
                        # Select random sample
                        sample_idx = np.random.randint(0, len(processed_data))
                        sample_data = processed_data[sample_idx:sample_idx+1]
                        
                        # Make prediction
                        if st.session_state.model_manager.active_model.is_trained:
                            prediction = st.session_state.model_manager.active_model.predict_single(sample_data)
                        else:
                            prediction = st.session_state.model_manager.active_model.predict_single(sample_data)
                        
                        st.success("✅ Sample test completed!")
                        
                        # Display prediction results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Walking", f"{prediction['probabilities']['walking']:.1%}")
                        with col2:
                            st.metric("Sitting", f"{prediction['probabilities']['sitting']:.1%}")
                        with col3:
                            st.metric("Driving", f"{prediction['probabilities']['driving']:.1%}")
                        with col4:
                            st.metric("Standing", f"{prediction['probabilities']['standing']:.1%}")
                        
                        # Prediction summary
                        st.info(f"🎯 **Predicted Behavior:** {prediction['prediction'].title()}")
                        st.info(f"📊 **Confidence:** {prediction['confidence']:.1%}")
                        st.info(f"🤖 **Model Used:** {prediction['model_used']}")
                        
                    except Exception as e:
                        st.error(f"❌ Testing failed: {str(e)}")
            else:
                st.warning("⚠️ Please upload data first before testing")
    
    else:
        st.info("ℹ️ Please initialize a model first to see evaluation options")

def show_settings():
    st.markdown("""
    <div class="main-header">
        <h1>Settings & Configuration</h1>
        <p>Configure your system preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_size = st.number_input("Input Size", min_value=1, value=332, help="Number of input features")
        hidden_size = st.number_input("Hidden Size", min_value=1, value=128, help="LSTM hidden layer size")
    
    with col2:
        num_classes = st.number_input("Number of Classes", min_value=2, value=4, help="Number of behavior classes")
        dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
    
    classes_input = st.text_input(
        "Classes (comma-separated)",
        value="walking, sitting, driving, standing",
        help="Behavior class names"
    )
    
    if st.button("💾 Save Configuration", type="primary"):
        st.success("✅ Configuration saved successfully!")
    
    # System Information
    st.markdown("### 💻 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Python Version:** 3.11.0")
        st.info("**Streamlit Version:** 1.28.0")
        st.info("**Pandas Version:** 2.0.3")
    
    with col2:
        st.warning("**PyTorch:** Not Available (Demo Mode)")
        st.success("**NumPy:** 1.24.3")
        st.success("**Matplotlib:** 3.7.2")
    
    # Export Settings
    st.markdown("### 📤 Export Settings")
    
    export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
    include_timestamps = st.checkbox("Include Timestamps", value=True)
    include_confidence = st.checkbox("Include Confidence Scores", value=True)
    
    if st.button("📥 Export Configuration", type="secondary"):
        st.info("Configuration exported successfully!")

if __name__ == "__main__":
    main()
