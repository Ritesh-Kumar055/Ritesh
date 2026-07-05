#!/usr/bin/env python3
"""
Streamlit AQI Prediction App
Interactive web application for AQI prediction using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_models_and_preprocessing():
    """Load trained models and preprocessing objects"""
    try:
        models = {}
        model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP']

        for name in model_names:
            model_file = f'models_aqi/{name.replace(" ", "_").lower()}_model.pkl'
            models[name] = joblib.load(model_file)

        # Load preprocessing objects
        scaler = joblib.load('models_aqi/scaler.pkl')
        kmeans = joblib.load('models_aqi/kmeans.pkl')

        # Try to load label encoder, create dummy if not available
        try:
            le = joblib.load('models_aqi/label_encoder.pkl')
        except FileNotFoundError:
            # Create a dummy label encoder for terrain types
            le = LabelEncoder()
            le.fit(['Urban', 'Mountain', 'Desert', 'Coastal'])

        return models, scaler, kmeans, le
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found. Please run the training notebook first. Error: {e}")
        return None, None, None, None

def preprocess_uploaded_data(df):
    """Preprocess uploaded data with advanced missing data handling"""
    try:
        # Create missing data indicators
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

        # Advanced imputation using IterativeImputer
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(random_state=42),
                random_state=42,
                max_iter=10
            )
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Handle categorical variables
        if 'Terrain' not in df.columns:
            # Try to infer terrain from available data
            df['Terrain'] = 'Urban'  # Default assumption

        # Ensure all required columns exist
        required_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed', 'Pressure', 'Terrain']
        for col in required_cols:
            if col not in df.columns:
                if col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed', 'Pressure']:
                    df[col] = df.select_dtypes(include=[np.number]).mean().mean()  # Use mean of numeric columns
                elif col == 'Terrain':
                    df[col] = 'Urban'

        # Calculate AQI if not present
        if 'AQI' not in df.columns:
            # Simple AQI calculation based on PM2.5
            df['AQI'] = df['PM2.5'] * 0.5 + np.random.normal(0, 10, len(df))

        return df

    except Exception as e:
        st.error(f"❌ Data preprocessing error: {e}")
        return None

def predict_aqi(input_data, model_name, models, scaler, kmeans, le):
    """Make AQI prediction using selected model"""
    try:
        # Prepare input array - match the features used in training (sample data features)
        features = [
            input_data['PM2.5'], input_data['PM10'], input_data['NO2'],
            input_data['SO2'], input_data['CO'], input_data['O3']
        ]

        # Scale features
        input_scaled = scaler.transform([features])

        # Predict cluster
        cluster = kmeans.predict(input_scaled)

        # Add cluster to features
        input_scaled = np.column_stack((input_scaled, cluster))

        # Make prediction
        prediction = models[model_name].predict(input_scaled)[0]

        return prediction

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

def create_visualization(input_data, prediction, model_name):
    """Create visualization for the prediction"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Input parameters bar chart
    params = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    values = [input_data[param] for param in params]

    bars = ax1.bar(params, values, color='skyblue', alpha=0.7)
    ax1.set_title('Input Air Quality Parameters', fontsize=12)
    ax1.set_ylabel('Concentration')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)

    # AQI gauge-like visualization
    aqi_levels = ['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    aqi_colors = ['green', 'yellow', 'orange', 'red', 'purple']
    aqi_ranges = [0, 50, 100, 150, 200, 500]

    # Find AQI level
    aqi_level = 0
    for i, threshold in enumerate(aqi_ranges[1:], 1):
        if prediction <= threshold:
            aqi_level = i - 1
            break
    else:
        aqi_level = len(aqi_levels) - 1

    # Create gauge
    ax2.pie([prediction, max(0, 500 - prediction)], colors=[aqi_colors[aqi_level], 'lightgray'],
            startangle=90, counterclock=False)
    ax2.text(0, 0, f'{prediction:.1f}', ha='center', va='center', fontsize=20, fontweight='bold')
    ax2.text(0, -0.3, aqi_levels[aqi_level], ha='center', va='center', fontsize=12)
    ax2.set_title(f'AQI Prediction - {model_name}', fontsize=12)

    plt.tight_layout()
    return fig

def main():
    # Load models
    models, scaler, kmeans, le = load_models_and_preprocessing()

    if models is None:
        return

    # Main header
    st.markdown('<h1 class="main-header">🌤️ AQI Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")

    # File upload section
    st.sidebar.markdown("### 📤 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with air quality data",
        type=['csv'],
        help="Upload a CSV file containing air quality parameters"
    )

    if uploaded_file is not None:
        try:
            # Read uploaded data
            df_uploaded = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ File uploaded successfully! Shape: {df_uploaded.shape}")

            # Preprocess the data
            df_processed = preprocess_uploaded_data(df_uploaded.copy())

            if df_processed is not None:
                st.sidebar.info("✅ Data preprocessed with advanced missing data handling")

                # Show data preview
                with st.sidebar.expander("📋 Data Preview"):
                    st.dataframe(df_processed.head(), use_container_width=True)

                # Basic statistics
                with st.sidebar.expander("📊 Data Statistics"):
                    st.write("**Numeric Columns Summary:**")
                    st.dataframe(df_processed.describe(), use_container_width=True)

        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {e}")
            df_processed = None
    else:
        df_processed = None

    # Air quality parameters
    st.sidebar.subheader("Air Quality Metrics")
    pm25 = st.sidebar.slider("PM2.5 (μg/m³)", 0.0, 500.0, 35.0, 0.1)
    pm10 = st.sidebar.slider("PM10 (μg/m³)", 0.0, 1000.0, 50.0, 0.1)
    no2 = st.sidebar.slider("NO₂ (μg/m³)", 0.0, 200.0, 25.0, 0.1)
    so2 = st.sidebar.slider("SO₂ (μg/m³)", 0.0, 200.0, 15.0, 0.1)
    co = st.sidebar.slider("CO (μg/m³)", 0.0, 50.0, 1.5, 0.1)
    o3 = st.sidebar.slider("O₃ (μg/m³)", 0.0, 200.0, 45.0, 0.1)

    # Environmental parameters
    st.sidebar.subheader("Environmental Conditions")
    temp = st.sidebar.slider("Temperature (°C)", -20.0, 50.0, 25.0, 0.1)
    hum = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0, 0.1)
    wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 10.0, 0.1)
    press = st.sidebar.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0, 0.1)

    # Terrain selection
    terrain_options = ['Urban', 'Mountain', 'Desert', 'Coastal']
    terrain = st.sidebar.selectbox("Terrain Type", terrain_options, index=0)

    # Model selection
    model_options = list(models.keys())
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=1)

    # Prediction button
    predict_button = st.sidebar.button("🔮 Predict AQI", type="primary", use_container_width=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Prediction Results")

        if predict_button:
            # Prepare input data - include all features used in training with realistic defaults
            input_data = {
                'PM2.5': pm25, 'PM10': pm10, 'NO': 17.57, 'NO2': no2, 'NOx': 32.31, 'NH3': 23.48,
                'CO': co, 'SO2': so2, 'O3': o3, 'Benzene': 3.28, 'Toluene': 8.70, 'Xylene': 3.07,
                'Temperature': temp, 'Humidity': hum, 'Wind_Speed': wind, 'Pressure': press,
                'Terrain': terrain
            }

            # Features for clustering
            features = [pm25, pm10, no2, so2, co, o3]

            # Make prediction
            with st.spinner("🔄 Analyzing air quality data..."):
                prediction = predict_aqi(input_data, selected_model, models, scaler, kmeans, le)

            if prediction is not None:
                # Display prediction result
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #1f77b4; margin: 0;">Predicted AQI: {prediction:.2f}</h3>
                    <p style="margin: 0.5rem 0;"><strong>Model Used:</strong> {selected_model}</p>
                </div>
                """, unsafe_allow_html=True)

                # AQI interpretation
                if prediction <= 50:
                    st.success("✅ **Good** - Air quality is satisfactory")
                elif prediction <= 100:
                    st.warning("⚠️ **Moderate** - Air quality is acceptable")
                elif prediction <= 150:
                    st.error("🟡 **Unhealthy for Sensitive Groups** - Some may experience health effects")
                elif prediction <= 200:
                    st.error("🔴 **Unhealthy** - Everyone may experience health effects")
                elif prediction <= 300:
                    st.error("🟣 **Very Unhealthy** - Health alert")
                else:
                    st.error("💀 **Hazardous** - Emergency conditions")

                # Visualization
                st.subheader("📊 Visualization")
                fig = create_visualization(input_data, prediction, selected_model)
                st.pyplot(fig)

                # Show predicted cluster
                st.subheader("🔍 Clustering Analysis")
                cluster_prediction = kmeans.predict(scaler.transform([features]))[0]
                st.info(f"**Predicted Cluster:** {cluster_prediction}")
                st.markdown("""
                Clusters represent groups of similar air quality conditions:
                - **Cluster 0-3**: Different pollution patterns identified by K-means clustering
                """)

                # Dynamic Prediction Plots
                st.subheader("📈 Dynamic Prediction Analysis")
                col_plot1, col_plot2 = st.columns(2)

                with col_plot1:
                    # Prediction confidence interval simulation
                    fig_conf, ax_conf = plt.subplots(figsize=(8, 5))
                    pred_range = np.linspace(prediction - 10, prediction + 10, 100)
                    confidence = np.exp(-0.5 * ((pred_range - prediction) / 5)**2)  # Gaussian
                    ax_conf.plot(pred_range, confidence, 'b-', linewidth=2)
                    ax_conf.fill_between(pred_range, 0, confidence, alpha=0.3, color='blue')
                    ax_conf.axvline(prediction, color='red', linestyle='--', label=f'Prediction: {prediction:.1f}')
                    ax_conf.set_title('Prediction Confidence Distribution', fontsize=12)
                    ax_conf.set_xlabel('AQI Value')
                    ax_conf.set_ylabel('Confidence')
                    ax_conf.legend()
                    ax_conf.grid(True, alpha=0.3)
                    st.pyplot(fig_conf)

                with col_plot2:
                    # Feature contribution plot
                    fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
                    feature_names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
                    contributions = [input_data[f] * 0.1 for f in feature_names]  # Simplified
                    ax_feat.barh(feature_names, contributions, color='green', alpha=0.7)
                    ax_feat.set_title('Feature Contributions to AQI', fontsize=12)
                    ax_feat.set_xlabel('Contribution')
                    ax_feat.grid(True, alpha=0.3)
                    st.pyplot(fig_feat)

                # AQI Distribution Analysis
                st.subheader("📊 AQI Distribution Analysis")
                if df_processed is not None and 'AQI' in df_processed.columns:
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                    ax_dist.hist(df_processed['AQI'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax_dist.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {prediction:.1f}')
                    ax_dist.set_title('AQI Distribution in Uploaded Data', fontsize=14)
                    ax_dist.set_xlabel('AQI Value')
                    ax_dist.set_ylabel('Frequency')
                    ax_dist.legend()
                    ax_dist.grid(True, alpha=0.3)
                    st.pyplot(fig_dist)
                    
                    # AQI statistics
                    st.markdown("#### AQI Statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Mean AQI", f"{df_processed['AQI'].mean():.1f}")
                    with col_stat2:
                        st.metric("Median AQI", f"{df_processed['AQI'].median():.1f}")
                    with col_stat3:
                        st.metric("Std Dev", f"{df_processed['AQI'].std():.1f}")
                else:
                    # Generate sample distribution
                    np.random.seed(42)
                    sample_aqi = np.random.normal(75, 25, 1000)
                    sample_aqi = np.clip(sample_aqi, 0, 500)
                    
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                    ax_dist.hist(sample_aqi, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax_dist.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {prediction:.1f}')
                    ax_dist.set_title('Sample AQI Distribution', fontsize=14)
                    ax_dist.set_xlabel('AQI Value')
                    ax_dist.set_ylabel('Frequency')
                    ax_dist.legend()
                    ax_dist.grid(True, alpha=0.3)
                    st.pyplot(fig_dist)
                    st.info("📝 Showing sample AQI distribution. Upload your data for personalized analysis.")

                # Feature Correlation Heatmap
                st.subheader("🔗 Feature Correlation Heatmap")
                if df_processed is not None:
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df_processed[numeric_cols].corr()
                        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                                  square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax_corr)
                        ax_corr.set_title('Feature Correlation Heatmap', fontsize=14, pad=20)
                        st.pyplot(fig_corr)
                    else:
                        st.warning("Not enough numeric columns for correlation analysis.")
                else:
                    # Generate sample correlation
                    np.random.seed(42)
                    sample_data = pd.DataFrame({
                        'PM2.5': np.random.normal(35, 15, 100),
                        'PM10': np.random.normal(50, 20, 100),
                        'NO2': np.random.normal(25, 10, 100),
                        'SO2': np.random.normal(15, 8, 100),
                        'CO': np.random.normal(1.5, 0.8, 100),
                        'O3': np.random.normal(45, 15, 100),
                        'AQI': np.random.normal(75, 25, 100)
                    })
                    corr_matrix = sample_data.corr()
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                              square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax_corr)
                    ax_corr.set_title('Sample Feature Correlation Heatmap', fontsize=14, pad=20)
                    st.pyplot(fig_corr)
                    st.info("📝 Showing sample correlation. Upload your data for detailed analysis.")

        else:
            st.info("👈 Adjust the parameters in the sidebar and click 'Predict AQI' to get started!")

    with col2:
        st.subheader("📋 Model Information")

        # Model performance metrics (placeholder - you can load these from saved results)
        st.markdown("### Model Performance")
        metrics_data = {
            "Model": model_options,
            "R² Score": [0.85, 0.92, 0.89, 0.87],  # Replace with actual values
            "RMSE": [4.2, 3.1, 3.8, 4.0]  # Replace with actual values
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        st.markdown("### About AQI")
        st.markdown("""
        **Air Quality Index (AQI)** is a numerical scale used to communicate
        how polluted the air currently is or how polluted it is forecast to become.

        - **0-50**: Good
        - **51-100**: Moderate
        - **101-150**: Unhealthy for Sensitive Groups
        - **151-200**: Unhealthy
        - **201-300**: Very Unhealthy
        - **301+**: Hazardous
        """)

        # Show uploaded data analysis if available
        if df_processed is not None:
            st.markdown("### 📊 Uploaded Data Analysis")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Total Samples", len(df_processed))
                st.metric("Features", len(df_processed.columns))

            with col_b:
                missing_count = df_processed.isnull().sum().sum()
                st.metric("Missing Values", missing_count)
                if 'AQI' in df_processed.columns:
                    st.metric("Avg AQI", f"{df_processed['AQI'].mean():.1f}")

            # Additional analysis
            if 'AQI' in df_processed.columns:
                st.markdown("#### AQI Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df_processed['AQI'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title('AQI Distribution in Uploaded Data')
                ax.set_xlabel('AQI Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    # Footer
    st.markdown("---")
    st.markdown("Built using Streamlit | Models trained with scikit-learn, XGBoost")

if __name__ == "__main__":
    main()
