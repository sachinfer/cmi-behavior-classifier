"""
Analytics module for data visualization and statistical analysis
"""

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

class DataVisualizer:
    """Handles data visualization and chart generation"""
    
    def __init__(self):
        self.style_setup()
    
    def style_setup(self):
        """Setup plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set figure size and DPI
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
    
    def create_correlation_heatmap(self, data, max_features=20):
        """Create correlation heatmap"""
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Limit features for visualization
        if len(numeric_data.columns) > max_features:
            numeric_data = numeric_data.iloc[:, :max_features]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_feature_distributions(self, data, features=None, max_features=6):
        """Create feature distribution plots"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if features is None:
            features = numeric_data.columns[:max_features]
        else:
            features = [f for f in features if f in numeric_data.columns]
        
        n_features = len(features)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Create histogram with KDE
            sns.histplot(data=numeric_data[feature], kde=True, ax=ax)
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(n_features, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_time_series_plot(self, data, time_column=None, features=None):
        """Create time series plots"""
        if time_column is None:
            # Try to find time-related columns
            time_candidates = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_candidates:
                time_column = time_candidates[0]
            else:
                # Create dummy time index
                time_column = 'index'
                data = data.reset_index()
        
        if features is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            features = numeric_cols[:5]  # Limit to 5 features
        
        # Create time series plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for feature in features:
            if feature in data.columns and feature != time_column:
                ax.plot(data[time_column], data[feature], label=feature, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_behavior_distribution(self, predictions, behaviors=None):
        """Create behavior distribution chart"""
        if behaviors is None:
            behaviors = ['walking', 'sitting', 'driving', 'standing']
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if isinstance(predictions, dict):
            # Single prediction
            values = [predictions.get(behavior, 0) for behavior in behaviors]
            labels = [behavior.title() for behavior in behaviors]
        else:
            # Multiple predictions
            values = [np.sum(predictions == behavior) for behavior in behaviors]
            labels = [behavior.title() for behavior in behaviors]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(behaviors)))
        
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        ax.set_title('Behavior Distribution', fontsize=16, pad=20)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_confusion_matrix(self, y_true, y_pred, labels=None):
        """Create confusion matrix visualization"""
        if labels is None:
            labels = ['walking', 'sitting', 'driving', 'standing']
        
        # Calculate confusion matrix
        cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_feature_importance_plot(self, feature_importance, feature_names=None, top_n=20):
        """Create feature importance plot"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_features = sorted_idx[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, feature_importance[top_features])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in top_features])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return img_str

class StatisticalAnalyzer:
    """Handles statistical analysis of data"""
    
    def __init__(self):
        pass
    
    def generate_summary_statistics(self, data):
        """Generate comprehensive summary statistics"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        summary = {
            'basic_stats': numeric_data.describe(),
            'missing_values': numeric_data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
        
        # Additional statistics
        summary['skewness'] = numeric_data.skew().to_dict()
        summary['kurtosis'] = numeric_data.kurtosis().to_dict()
        summary['correlation_with_target'] = {}
        
        return summary
    
    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """Detect outliers in numeric data"""
        numeric_data = data.select_dtypes(include=[np.number])
        outliers = {}
        
        for column in numeric_data.columns:
            if method == 'iqr':
                Q1 = numeric_data[column].quantile(0.25)
                Q3 = numeric_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers[column] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': np.sum((numeric_data[column] < lower_bound) | 
                                          (numeric_data[column] > upper_bound)),
                    'outlier_indices': np.where((numeric_data[column] < lower_bound) | 
                                              (numeric_data[column] > upper_bound))[0]
                }
        
        return outliers
    
    def analyze_data_quality(self, data):
        """Analyze overall data quality"""
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Calculate percentages
        quality_report['missing_percentage'] = (quality_report['missing_values'] / 
                                              (quality_report['total_rows'] * quality_report['total_columns'])) * 100
        
        quality_report['duplicate_percentage'] = (quality_report['duplicate_rows'] / 
                                                quality_report['total_rows']) * 100
        
        # Quality score (0-100)
        quality_score = 100
        quality_score -= quality_report['missing_percentage'] * 0.5
        quality_score -= quality_report['duplicate_percentage'] * 0.3
        quality_score = max(0, quality_score)
        
        quality_report['quality_score'] = quality_score
        
        return quality_report
    
    def generate_correlation_analysis(self, data, target_column=None):
        """Generate correlation analysis"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_column and target_column in numeric_data.columns:
            correlations = numeric_data.corr()[target_column].sort_values(ascending=False)
            return correlations
        else:
            return numeric_data.corr()
    
    def perform_statistical_tests(self, data, feature1, feature2):
        """Perform statistical tests between two features"""
        if feature1 not in data.columns or feature2 not in data.columns:
            return None
        
        from scipy import stats
        
        # Ensure numeric data
        if not (np.issubdtype(data[feature1].dtype, np.number) and 
                np.issubdtype(data[feature2].dtype, np.number)):
            return None
        
        # Remove NaN values
        clean_data = data[[feature1, feature2]].dropna()
        
        if len(clean_data) < 3:
            return None
        
        # Perform tests
        tests = {}
        
        # Correlation test
        correlation, p_value = stats.pearsonr(clean_data[feature1], clean_data[feature2])
        tests['pearson_correlation'] = {'correlation': correlation, 'p_value': p_value}
        
        # T-test (if applicable)
        if len(clean_data[feature1].unique()) == 2:  # Binary feature
            groups = [clean_data[clean_data[feature1] == val][feature2] for val in clean_data[feature1].unique()]
            if len(groups[0]) > 0 and len(groups[1]) > 0:
                t_stat, t_p_value = stats.ttest_ind(groups[0], groups[1])
                tests['t_test'] = {'t_statistic': t_stat, 'p_value': t_p_value}
        
        return tests
