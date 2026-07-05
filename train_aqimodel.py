#!/usr/bin/env python
# coding: utf-8

# # AQI Model Training with Comprehensive Visualizations
# This notebook trains AQI prediction models with k-means clustering and extensive data visualizations.

# In[ ]:


import pandas as pd   
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


# In[ ]:


def load_data_from_folder(data_folder='data'):
    """
    Load data from CSV files in the data folder
    """
    if os.path.exists(data_folder):
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if csv_files:
            # Load the first CSV file found
            file_path = os.path.join(data_folder, csv_files[0])
            df = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}")
            print(f"Dataset shape: {df.shape}")
            return df

    # Fallback to sample data if no data folder or CSV files
    print("No data folder found, generating sample data...")
    return generate_sample_data()

def generate_sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'PM2.5': np.random.normal(35, 15, n_samples),
        'PM10': np.random.normal(50, 20, n_samples),
        'NO2': np.random.normal(25, 10, n_samples),
        'SO2': np.random.normal(15, 8, n_samples),
        'CO': np.random.normal(1.5, 0.8, n_samples),
        'O3': np.random.normal(45, 15, n_samples),
        'Temperature': np.random.normal(25, 8, n_samples),
        'Humidity': np.random.normal(60, 20, n_samples),
        'Terrain': np.random.choice(['Urban', 'Mountain', 'Desert', 'Coastal'], n_samples),
        'Wind_Speed': np.random.normal(10, 5, n_samples),
        'Pressure': np.random.normal(1013, 50, n_samples)
    }

    df = pd.DataFrame(data)
    df['AQI'] = (0.3 * df['PM2.5'] + 0.2 * df['PM10'] + 0.15 * df['NO2'] + 
                0.1 * df['SO2'] + 0.1 * df['CO'] + 0.15 * df['O3'] +
                np.random.normal(0, 5, n_samples))

    return df

# Load data
df = load_data_from_folder()
print("\nFirst 5 rows:")
df.head()


# In[ ]:


# Basic data exploration
print("Dataset Info:")
print(df.info())
print("\n" + "="*50)
print("Statistical Summary:")
print(df.describe())
print("\n" + "="*50)
print("Missing Values:")
print(df.isnull().sum())


# In[ ]:


# Comprehensive Data Visualizations

# 1. Correlation Heatmap
plt.figure(figsize=(14, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


# In[ ]:


# 2. Distribution of AQI
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# AQI Distribution
sns.histplot(df['AQI'], bins=30, kde=True, ax=ax1, color='skyblue')
ax1.set_title('AQI Distribution', fontsize=14)
ax1.set_xlabel('AQI Value')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# AQI by Terrain (Box Plot)
if 'Terrain' in df.columns:
    sns.boxplot(x='Terrain', y='AQI', data=df, ax=ax2, palette='Set3')
    ax2.set_title('AQI Distribution by Terrain', fontsize=14)
    ax2.set_xlabel('Terrain Type')
    ax2.set_ylabel('AQI Value')
    ax2.tick_params(axis='x', rotation=45)
else:
    ax2.text(0.5, 0.5, 'No Terrain column found', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('AQI by Terrain (N/A)', fontsize=14)

# PM2.5 vs AQI Scatter
sns.scatterplot(x='PM2.5', y='AQI', data=df, ax=ax3, alpha=0.6, color='red')
ax3.set_title('PM2.5 vs AQI', fontsize=14)
ax3.set_xlabel('PM2.5 Concentration')
ax3.set_ylabel('AQI Value')
ax3.grid(True, alpha=0.3)

# PM10 vs AQI Scatter (using available feature)
if 'PM10' in df.columns:
    sns.scatterplot(x='PM10', y='AQI', data=df, ax=ax4, alpha=0.6, color='green')
    ax4.set_title('PM10 vs AQI', fontsize=14)
    ax4.set_xlabel('PM10 Concentration')
    ax4.set_ylabel('AQI Value')
else:
    ax4.text(0.5, 0.5, 'PM10 column not found', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('PM10 vs AQI (N/A)', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# 3. Box Plots for All Features
numeric_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed', 'Pressure']
available_features = [col for col in numeric_features if col in df.columns]

n_features = len(available_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
axes = axes.flatten()

for i, feature in enumerate(available_features):
    sns.boxplot(y=df[feature], ax=axes[i], color='lightblue')
    axes[i].set_title(f'{feature} Distribution', fontsize=12)
    axes[i].set_ylabel(feature)
    axes[i].grid(True, alpha=0.3)

# Hide empty subplots
for i in range(n_features, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Box Plots of All Features', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


# In[ ]:


# Data Preprocessing
print("Starting Data Preprocessing...")

# Handle missing values
df = df.dropna()
print(f"After dropping missing values: {df.shape}")

# Encode categorical variables
le = LabelEncoder()
if 'Terrain' in df.columns:
    le.fit(df['Terrain'])  # Fit before transform
    df['Terrain_encoded'] = le.transform(df['Terrain'])
    # Use only the features that are actually in the sample data
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Terrain_encoded']
else:
    # Use only the features that are actually in the sample data
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# Filter features to only those available in the dataset
features = [f for f in features if f in df.columns]

X = df[features]
y = df['AQI']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
train_clusters = kmeans.predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# Add cluster labels as a new feature
X_train_scaled = np.column_stack((X_train_scaled, train_clusters))
X_test_scaled = np.column_stack((X_test_scaled, test_clusters))

print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
print(f"Clusters created: {len(np.unique(train_clusters))}")


# In[ ]:


# K-Means Clustering Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Cluster Distribution
cluster_counts = pd.Series(train_clusters).value_counts().sort_index()
bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_title('K-Means Cluster Distribution', fontsize=14)
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Number of Samples')
ax1.set_xticks(range(len(cluster_counts)))
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, count in zip(bars, cluster_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom')

# Scatter plot of clusters (first two features)
scatter = ax2.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                     c=train_clusters, cmap='viridis', alpha=0.6, s=50)
ax2.set_title('K-Means Clustering Visualization', fontsize=14)
ax2.set_xlabel('Scaled Feature 1')
ax2.set_ylabel('Scaled Feature 2')
plt.colorbar(scatter, ax=ax2, label='Cluster')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
}

results = {}
predictions = {}

print("Training Models...")
print("=" * 60)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    predictions[name] = y_pred

    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("-" * 40)


# In[ ]:


# Model Performance Comparison
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['rmse'] for m in results],
    'MAE': [results[m]['mae'] for m in results],
    'R2': [results[m]['r2'] for m in results]
})

# Find best model
best_model_name = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
best_model = results[best_model_name]

print("MODEL PERFORMANCE SUMMARY")
print("=" * 50)
print(metrics_df.to_string(index=False))
print("\n" + "=" * 50)
print(f"Best Model: {best_model_name}")
print(f"RMSE: {best_model['rmse']:.4f}")
print(f"MAE: {best_model['mae']:.4f}")
print(f"R² Score: {best_model['r2']:.4f}")


# In[ ]:


# Model Comparison Visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

# RMSE Comparison
bars1 = ax1.bar(metrics_df['Model'], metrics_df['RMSE'], color='skyblue', alpha=0.8)
ax1.set_title('Model RMSE Comparison', fontsize=14)
ax1.set_ylabel('RMSE (Lower is Better)')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

# MAE Comparison
bars2 = ax2.bar(metrics_df['Model'], metrics_df['MAE'], color='lightgreen', alpha=0.8)
ax2.set_title('Model MAE Comparison', fontsize=14)
ax2.set_ylabel('MAE (Lower is Better)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

# R² Score Comparison
bars3 = ax3.bar(metrics_df['Model'], metrics_df['R2'], color='coral', alpha=0.8)
ax3.set_title('Model R² Score Comparison', fontsize=14)
ax3.set_ylabel('R² Score (Higher is Better)')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.4f}', ha='center', va='bottom')

# Overall Performance Radar Chart
categories = ['RMSE', 'MAE', 'R2']
fig_radar, ax4 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Normalize the metrics for radar chart
normalized_metrics = metrics_df.copy()
normalized_metrics['RMSE'] = 1 - (normalized_metrics['RMSE'] - normalized_metrics['RMSE'].min()) / (normalized_metrics['RMSE'].max() - normalized_metrics['RMSE'].min())
normalized_metrics['MAE'] = 1 - (normalized_metrics['MAE'] - normalized_metrics['MAE'].min()) / (normalized_metrics['MAE'].max() - normalized_metrics['MAE'].min())
normalized_metrics['R2'] = (normalized_metrics['R2'] - normalized_metrics['R2'].min()) / (normalized_metrics['R2'].max() - normalized_metrics['R2'].min())

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i, model in enumerate(normalized_metrics['Model']):
    values = normalized_metrics.loc[i, categories].values.tolist()
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2, label=model)
    ax4.fill(angles, values, alpha=0.25)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_ylim(0, 1)
ax4.set_title('Model Performance Comparison (Normalized)', size=14, fontweight='bold')
ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
ax4.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:


# Actual vs Predicted Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]

    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add metrics as text
    r2 = results[name]['r2']
    rmse = results[name]['rmse']
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_title(f'{name}: Actual vs Predicted', fontsize=14)
    ax.set_xlabel('Actual AQI')
    ax.set_ylabel('Predicted AQI')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Actual vs Predicted AQI Values', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


# In[ ]:


# Residual Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]

    residuals = y_test - y_pred

    # Residual plot
    ax.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

    ax.set_title(f'{name}: Residual Plot', fontsize=14)
    ax.set_xlabel('Predicted AQI')
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)

plt.suptitle('Residual Analysis for All Models', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


# In[ ]:


# Feature Importance for Tree-based Models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Feature Importance
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    rf_importance = rf_model.feature_importances_
    features_with_cluster = features + ['Cluster']

    rf_df = pd.DataFrame({'feature': features_with_cluster, 'importance': rf_importance})
    rf_df = rf_df.sort_values('importance', ascending=True)

    ax1.barh(rf_df['feature'], rf_df['importance'], color='forestgreen', alpha=0.8)
    ax1.set_title('Random Forest Feature Importance', fontsize=14)
    ax1.set_xlabel('Importance')
    ax1.grid(True, alpha=0.3)

# XGBoost Feature Importance
if 'XGBoost' in results:
    xgb_model = results['XGBoost']['model']
    xgb_importance = xgb_model.feature_importances_

    xgb_df = pd.DataFrame({'feature': features_with_cluster, 'importance': xgb_importance})
    xgb_df = xgb_df.sort_values('importance', ascending=True)

    ax2.barh(xgb_df['feature'], xgb_df['importance'], color='darkorange', alpha=0.8)
    ax2.set_title('XGBoost Feature Importance', fontsize=14)
    ax2.set_xlabel('Importance')
    ax2.grid(True, alpha=0.3)

plt.suptitle('Feature Importance Analysis', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


# In[ ]:


# Save Models and Preprocessing Objects
os.makedirs('models_aqi', exist_ok=True)

for name, res in results.items():
    model_path = f"models_aqi/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(res['model'], model_path)
    print(f"✓ Saved {name} model to {model_path}")

# Save preprocessing objects
joblib.dump(scaler, 'models_aqi/scaler.pkl')
joblib.dump(kmeans, 'models_aqi/kmeans.pkl')
if 'le' in locals():
    joblib.dump(le, 'models_aqi/label_encoder.pkl')

print("\n✓ All models and preprocessing objects saved!")
print("\n🎉 Training completed successfully!")
print(f"📊 Best performing model: {best_model_name}")
print(f"📈 Best R² Score: {best_model['r2']:.4f}")
print("\n🚀 Ready for deployment with Streamlit app!")

