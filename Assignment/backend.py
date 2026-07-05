# backend.py
# Digital Twin + CNN-LSTM forecasting pipeline (single-file)
# Run: python backend.py
# Make sure 'powerconsumption.csv' is in the same folder (or update DATA_PATH)

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# sklearn imports with graceful failure message
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except Exception as e:
    raise SystemExit("scikit-learn is required but could not be imported: " + str(e) +
                     "\nInstall it with: pip install scikit-learn")

# joblib for saving scalers/models
try:
    import joblib
except Exception:
    joblib = None
    warnings.warn("joblib not available; model save/load will be skipped. Install with: pip install joblib")

# TensorFlow / Keras
try:
    import tensorflow as tf
    # reference Keras objects via the top-level tf.keras namespace to avoid direct submodule imports
    keras = tf.keras
    Model = keras.models.Model
    Input = keras.layers.Input
    Conv1D = keras.layers.Conv1D
    MaxPooling1D = keras.layers.MaxPooling1D
    LSTM = keras.layers.LSTM
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    EarlyStopping = keras.callbacks.EarlyStopping
    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
except Exception as e:
    raise SystemExit("TensorFlow/Keras is required but could not be imported: " + str(e) +
                     "\nInstall it with: pip install tensorflow")

# Optional: SHAP & seaborn & statsmodels
try:
    import importlib
    shap = importlib.import_module('shap')
except Exception:
    shap = None

try:
    import importlib
    sns = importlib.import_module('seaborn')
except Exception:
    sns = None

try:
    import importlib
    _sm = importlib.import_module('statsmodels.graphics.tsaplots')
    plot_acf = getattr(_sm, 'plot_acf', None)
    plot_pacf = getattr(_sm, 'plot_pacf', None)
except Exception:
    plot_acf = None
    plot_pacf = None

    import matplotlib
matplotlib.use('Agg')   # force non-interactive backend for PNG export


# -------------------------
# Configuration
# -------------------------
DATA_PATH = 'powerconsumption.csv'   # name of CSV in current directory
OUT_DIR = 'outputs'
MODEL_DIR = 'models'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model & training hyperparameters (tune as needed)
SEQ_LEN = 144    # history length (10-min freq -> 144 ~ 24 hours)
HORIZON = 6      # predict next 6 steps (60 minutes)
EPOCHS = 3       # reduced for faster runs
es = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

BATCH_SIZE = 128  # larger batch for faster training
RANDOM_STATE = 42

# Speed caps (automatic performance heuristics)
RF_MAX_TRAIN_SAMPLES = 20000    # subsample RF training if dataset larger
SHAP_MAX_SAMPLES = 1000         # sample for SHAP
SEQ_MAX_SAMPLES = 20000         # cap sequence samples (keep most recent)

# -------------------------
# Utility functions
# -------------------------
def map_columns(df):
    """
    Map likely column names to canonical names:
    Datetime, Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows,
    PowerZone1, PowerZone2, PowerZone3
    """
    cols = {c.lower().strip(): c for c in df.columns}
    def find(keys):
        for k in keys:
            for c in cols:
                if k in c:
                    return cols[c]
        return None

    mapping = {}
    mapping['Datetime'] = find(['datetime', 'date', 'time'])
    mapping['Temperature'] = find(['temp', 'temperature'])
    mapping['Humidity'] = find(['humid'])
    mapping['WindSpeed'] = find(['wind'])
    mapping['GeneralDiffuseFlows'] = find(['general diffuse']) or find(['generaldiffuse'])
    mapping['DiffuseFlows'] = find(['diffuse flows', 'diffuseflows', 'diffuse'])
    mapping['PowerZone1'] = find(['zone1', 'powerconsumption_zone1', 'power consumption', 'powerconsumption'])
    mapping['PowerZone2'] = find(['zone2', 'powerconsumption_zone2', 'power contumption', 'zone 2'])
    mapping['PowerZone3'] = find(['zone3', 'powerconsumption_zone3', 'power consumption zone3'])

    new = pd.DataFrame()
    for k, v in mapping.items():
        if v and v in df.columns:
            new[k] = df[v]
        else:
            new[k] = np.nan
    return new

def safe_to_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(',',''), errors='coerce')

# -------------------------
# Load data
# -------------------------
print('Loading data...')
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"CSV file not found: {DATA_PATH}. Please place it alongside this script or update DATA_PATH.")

raw = pd.read_csv(DATA_PATH, encoding='utf-8', low_memory=False)
print('Original columns:', list(raw.columns[:20]))

# Try to detect datetime in first column or a named datetime column
print("Checking Datetime column...")

# If a clearly-named datetime column exists, use it; otherwise try first column
datetime_col = None
for c in raw.columns:
    if 'date' in c.lower() or 'time' in c.lower():
        datetime_col = c
        break
if datetime_col is None:
    datetime_col = raw.columns[0]

# Attempt parse
first_col_parsed = pd.to_datetime(raw[datetime_col], errors='coerce', dayfirst=False)
if first_col_parsed.notna().sum() < 3:
    # try a few initial rows as header shift scenario
    found = None
    for i in range(0, 10):
        try:
            test = pd.to_datetime(raw.iloc[i, 0], errors='coerce')
            if not pd.isna(test):
                found = i
                break
        except Exception:
            pass
    if found is not None and found > 0:
        # reload skipping rows
        raw = pd.read_csv(DATA_PATH, skiprows=found, low_memory=False)
        print(f"Reloaded CSV skipping first {found} rows to find header")
        datetime_col = raw.columns[0]
        first_col_parsed = pd.to_datetime(raw[datetime_col], errors='coerce', dayfirst=False)

if first_col_parsed.notna().sum() < 3:
    print("Warning: Could not parse many datetimes in the detected column. Proceeding but check your CSV.")
else:
    raw[datetime_col] = first_col_parsed

# Map columns to canonical names
df = map_columns(raw)
print('Mapped columns:', df.columns.tolist())

# If Datetime still NaN, fallback to raw first column
if df['Datetime'].isna().all():
    df['Datetime'] = raw.iloc[:, 0]

# Ensure datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df = df[~df['Datetime'].isna()].copy()

# Convert numeric columns
for c in ['PowerZone1','PowerZone2','PowerZone3']:
    if c in df.columns:
        df[c] = safe_to_numeric(df[c])
for c in ['Temperature','Humidity','WindSpeed','GeneralDiffuseFlows','DiffuseFlows']:
    if c in df.columns:
        df[c] = safe_to_numeric(df[c])

# Sort & index
df = df.sort_values('Datetime').reset_index(drop=True)
print('Data range:', df['Datetime'].min(), 'to', df['Datetime'].max())

# Set index
df.set_index('Datetime', inplace=True)
df = df[~df.index.duplicated(keep='first')]

# Infer frequency; resample to 10T if unknown
inferred = None
try:
    inferred = pd.infer_freq(df.index[:100])
except Exception:
    inferred = None
print('Inferred freq (first100):', inferred)
if inferred is None:
    df = df.resample('10T').mean()
    df = df.ffill().bfill()
else:
    df = df.asfreq(inferred).interpolate().ffill().bfill()

# Quick numeric cleaning
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].interpolate().ffill().bfill()

# Snapshot
df.head().to_csv(os.path.join(OUT_DIR, 'data_snapshot.csv'))

# -------------------------
# EDA & plots
# -------------------------
print('Creating exploratory plots...')
plt.figure(figsize=(12,6))
for col in ['PowerZone1','PowerZone2','PowerZone3']:
    if col in df.columns:
        plt.plot(df.index, df[col], label=col)
plt.title('Power Consumption Over Time')
plt.xlabel('Datetime')
plt.ylabel('Power Consumption')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'timeseries_zones.png'))
plt.close()

if 'Temperature' in df.columns and 'PowerZone1' in df.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(df['Temperature'], df['PowerZone1'], s=2)
    plt.title('Temperature vs PowerConsumption (Zone1)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Power')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'temp_vs_zone1.png'))
    plt.close()

if sns is not None:
    try:
        plt.figure(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f')
        plt.title('Feature Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'correlation.png'))
        plt.close()
    except Exception:
        pass

# -------------------------
# Anomaly detection
# -------------------------
print('Detecting anomalies...')
iso_fit_cols = [c for c in ['PowerZone1','PowerZone2','PowerZone3'] if c in df.columns]
if len(iso_fit_cols) >= 1:
    iso = IsolationForest(contamination=0.001, random_state=RANDOM_STATE)
    iso.fit(df[iso_fit_cols])
    df['anomaly_score'] = iso.decision_function(df[iso_fit_cols])
    df['anomaly'] = iso.predict(df[iso_fit_cols])
    anomalies = df[df['anomaly'] == -1]
    print('Detected anomalies:', len(anomalies))
    anomalies.head().to_csv(os.path.join(OUT_DIR, 'anomalies.csv'))
    # plot anomalies
    try:
        plt.figure(figsize=(12,4))
        plt.plot(df.index, df['PowerZone1'], label='Zone1')
        if not anomalies.empty:
            plt.scatter(anomalies.index, anomalies['PowerZone1'], c='r', s=8, label='anomaly')
        plt.title('Zone1 with Anomalies')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'anomalies_zone1.png'))
        plt.close()
    except Exception:
        pass

# -------------------------
# Digital Twin (simple HVAC heuristic)
# -------------------------
print('Running Digital Twin simulation...')
cooling_setpoint = 26.0
heating_setpoint = 18.0
hvac_sens_zone = {'PowerZone1': 180.0, 'PowerZone2': 150.0, 'PowerZone3': 160.0}

if 'Temperature' in df.columns:
    temp = df['Temperature']
    for z in list(hvac_sens_zone.keys()):
        if z in df.columns:
            cool_extra = np.maximum(0, temp - cooling_setpoint) * hvac_sens_zone[z]
            heat_extra = np.maximum(0, heating_setpoint - temp) * (hvac_sens_zone[z] * 0.4)
            hvac_term = cool_extra + heat_extra
            df[z + '_DT_sim'] = df[z].rolling(window=6, min_periods=1).mean() + hvac_term

    dt_cols = [c for c in df.columns if c.endswith('_DT_sim')]
    if dt_cols:
        df[dt_cols].head().to_csv(os.path.join(OUT_DIR, 'dt_sim_snapshot.csv'))
        # Plot sample
        try:
            z0 = 'PowerZone1'
            if z0 in df.columns and z0 + '_DT_sim' in df.columns:
                plt.figure(figsize=(10,4))
                plt.plot(df.index[:144], df[z0].iloc[:144], label='Original')
                plt.plot(df.index[:144], df[z0 + '_DT_sim'].iloc[:144], label='DT_sim')
                plt.title('Original vs DT_sim (first 24h)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, 'dt_vs_original_zone1.png'))
                plt.close()
        except Exception:
            pass

# -------------------------
# Feature engineering
# -------------------------
print('Feature engineering...')
features = []
if 'Temperature' in df.columns:
    df['temp_diff'] = df['Temperature'].diff().fillna(0)
    df['temp_roll_mean_3'] = df['Temperature'].rolling(window=3, min_periods=1).mean()
    features += ['Temperature', 'temp_diff', 'temp_roll_mean_3']

# calendar features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
features += ['hour', 'dayofweek']

# target selection (prefer PowerZone1)
target = 'PowerZone1'
if target not in df.columns:
    power_cols = [c for c in ['PowerZone1','PowerZone2','PowerZone3'] if c in df.columns]
    if len(power_cols) == 0:
        raise SystemExit("No power consumption columns found in dataset.")
    target = power_cols[0]

# lags and rolling stats
for lag in [1,6,12,24,48]:
    df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    features.append(f'{target}_lag_{lag}')
for win in [6,12,24]:
    df[f'{target}_roll_mean_{win}'] = df[target].rolling(window=win, min_periods=1).mean()
    features.append(f'{target}_roll_mean_{win}')

# include DT features if present
dt_cols = [c for c in df.columns if c.endswith('_DT_sim')]
features += dt_cols

# drop NaNs (due to shifts)
df_ml = df.copy().dropna(subset=features + [target])
print('ML dataset size:', df_ml.shape)

# -------------------------
# Train / test split (time-based)
# -------------------------
train_size = int(len(df_ml) * 0.8)
train = df_ml.iloc[:train_size]
test = df_ml.iloc[train_size:]

X_train = train[features].values
y_train = train[target].values
X_test = test[features].values
y_test = test[target].values

# scale features (for RF baseline and also for sequence pipeline later)
sc_global = StandardScaler()
X_train_s = sc_global.fit_transform(X_train)
X_test_s = sc_global.transform(X_test)
if joblib:
    joblib.dump(sc_global, os.path.join(MODEL_DIR, 'sc_global.pkl'))

# -------------------------
# RandomForest baseline + SHAP
# -------------------------
print('Training RandomForest baseline...')
# Subsample RF training if dataset very large to speed up
if X_train_s.shape[0] > RF_MAX_TRAIN_SAMPLES:
    rng = np.random.RandomState(RANDOM_STATE)
    idx_sub = rng.choice(X_train_s.shape[0], RF_MAX_TRAIN_SAMPLES, replace=False)
    X_rf_train = X_train_s[idx_sub]
    y_rf_train = y_train[idx_sub]
    print(f'RF training: subsampled {X_rf_train.shape[0]} rows from {X_train_s.shape[0]} for speed.')
else:
    X_rf_train = X_train_s
    y_rf_train = y_train

# reduce n_estimators for speed
rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_rf_train, y_rf_train)
if joblib:
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_baseline.pkl'))

y_pred_rf = rf.predict(X_test_s)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'RF MAE: {rf_mae:.3f}, RF RMSE: {rf_rmse:.3f}')

# SHAP summary (optional) with sampling to avoid huge runtimes
if shap is not None:
    try:
        print("Running SHAP (sampled)...")
        explainer = shap.TreeExplainer(rf)
        # sample for SHAP if necessary
        if X_test_s.shape[0] > SHAP_MAX_SAMPLES:
            rng = np.random.RandomState(RANDOM_STATE)
            idx_shap = rng.choice(X_test_s.shape[0], SHAP_MAX_SAMPLES, replace=False)
            X_shap = X_test_s[idx_shap]
            print(f'Computing SHAP on {len(X_shap)} samples (of {X_test_s.shape[0]}) for speed.')
        else:
            X_shap = X_test_s

        shap_vals = explainer.shap_values(X_shap)
        print("SHAP values computed:", np.shape(shap_vals))

        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_vals, pd.DataFrame(X_shap, columns=features), show=False)
        plt.tight_layout()

        shap_path = os.path.join(OUT_DIR, 'shap_summary.png')
        plt.savefig(shap_path)
        plt.close()

        print("SHAP saved to:", shap_path)

    except Exception as e:
        print("❌ SHAP failed with error:", e)
else:
    print("SHAP not installed.")

# -------------------------
# Sequence creation for CNN+LSTM (correct alignment)
# -------------------------
print('Preparing sequence data for CNN+LSTM...')

def create_sequences(data_df, feature_cols, target_col, seq_len=SEQ_LEN, horizon=HORIZON):
    """
    Create sequence samples:
    - X shape: (n_samples, seq_len, n_features)
    - y shape: (n_samples, horizon)
    Alignment: sample i predicts timestamps starting at data_df.index[seq_len + i]
    """
    arr = data_df[feature_cols].values
    targ = data_df[target_col].values
    N = len(data_df)
    Xs, ys = [], []
    for i in range(seq_len, N - horizon + 1):
        Xs.append(arr[i - seq_len:i])
        ys.append(targ[i:i + horizon])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(df_ml, features, target, seq_len=SEQ_LEN, horizon=HORIZON)
print('Sequence shapes before capping:', X_seq.shape, y_seq.shape)

# cap sequence samples to keep memory/time bounded (keep most recent samples)
if X_seq.shape[0] > SEQ_MAX_SAMPLES:
    keep = SEQ_MAX_SAMPLES
    X_seq = X_seq[-keep:]
    y_seq = y_seq[-keep:]
    print(f'Capped sequences to most recent {keep} samples for speed.')

print('Sequence shapes:', X_seq.shape, y_seq.shape)

# sequence-level train/test split (time ordered)
seq_count = X_seq.shape[0]
split_idx = int(seq_count * 0.8)
X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
print('Sequence train/test:', X_seq_train.shape, X_seq_test.shape)

# scale X (sequence-aware): fit on flattened training features then reshape
nsamples, nsteps, nfeatures = X_seq_train.shape
X_flat_train = X_seq_train.reshape(-1, nfeatures)
X_flat_test = X_seq_test.reshape(-1, nfeatures)

sc_X = StandardScaler()
X_flat_train_s = sc_X.fit_transform(X_flat_train)
X_flat_test_s = sc_X.transform(X_flat_test)

X_seq_train_s = X_flat_train_s.reshape(nsamples, nsteps, nfeatures)
X_seq_test_s = X_flat_test_s.reshape(X_seq_test.shape)
if joblib:
    joblib.dump(sc_X, os.path.join(MODEL_DIR, 'sc_X.pkl'))

# scale y (multi-step) using StandardScaler on flattened y_train
y_flat_train = y_seq_train.reshape(-1, 1)
sc_y = StandardScaler()
y_flat_train_s = sc_y.fit_transform(y_flat_train)
y_seq_train_s = y_flat_train_s.reshape(y_seq_train.shape)
if joblib:
    joblib.dump(sc_y, os.path.join(MODEL_DIR, 'sc_y.pkl'))

# -------------------------
# Build improved (but smaller) CNN + stacked LSTM model for speed
# -------------------------
print('Building CNN+LSTM model...')
input_layer = Input(shape=(nsteps, nfeatures))
# reduced model size for faster training
x = Conv1D(filters=32, kernel_size=24, activation='relu', padding='same')(input_layer)
x = Conv1D(filters=32, kernel_size=12, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)
x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=False)(x)
x = Dropout(0.2)(x)
output = Dense(HORIZON, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Callbacks
es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
cp = ModelCheckpoint(os.path.join(MODEL_DIR, 'cnn_lstm_best.h5'), save_best_only=True, monitor='val_loss', verbose=1)

# Train model
print('Training model...')
# For validation during training we can provide scaled y test; Keras expects same shape
# We'll create scaled y_seq_test_s for validation
y_seq_test_s = sc_y.transform(y_seq_test.reshape(-1,1)).reshape(y_seq_test.shape)

history = model.fit(
    X_seq_train_s, y_seq_train_s,
    validation_data=(X_seq_test_s, y_seq_test_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, rlrp, cp],
    verbose=1
)

# save training curve
plt.figure(figsize=(8,4))
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend()
plt.title('Training Loss')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_loss.png'))
plt.close()

# -------------------------
# Predict and inverse-transform
# -------------------------
print('Predicting on test set...')
y_pred_s = model.predict(X_seq_test_s)   # scaled predictions
y_pred_flat_s = y_pred_s.reshape(-1,1)
y_pred_flat = sc_y.inverse_transform(y_pred_flat_s)
y_pred = y_pred_flat.reshape(y_pred_s.shape)

# true test values (original scale)
y_test_true = y_seq_test

# metrics per horizon & overall
mae_per_h = []
rmse_per_h = []
for h in range(HORIZON):
    mae_h = mean_absolute_error(y_test_true[:,h], y_pred[:,h])
    rmse_h = np.sqrt(mean_squared_error(y_test_true[:,h], y_pred[:,h]))
    mae_per_h.append(mae_h)
    rmse_per_h.append(rmse_h)
    print(f'Horizon {h+1}: MAE = {mae_h:.3f}, RMSE = {rmse_h:.3f}')

mae_overall = mean_absolute_error(y_test_true.flatten(), y_pred.flatten())
rmse_overall = np.sqrt(mean_squared_error(y_test_true.flatten(), y_pred.flatten()))
print(f'Overall MAE (all horizons): {mae_overall:.3f}, Overall RMSE: {rmse_overall:.3f}')

# -------------------------
# Save forecasts aligned with timestamps
# -------------------------
print('Saving forecast CSV...')
n_test = X_seq_test.shape[0]
pred_rows = []
for j in range(n_test):
    global_seq_idx = split_idx + j  # index in full sequence array
    # timestamp for first horizon of this sample:
    # The i'th sample corresponds to predictions starting at df_ml.index[SEQ_LEN + i]
    ts_index = SEQ_LEN + global_seq_idx
    # safe bounding
    if ts_index >= len(df_ml.index):
        ts_index = len(df_ml.index) - 1
    timestamp = df_ml.index[ts_index]
    row = {'timestamp': timestamp}
    for h in range(HORIZON):
        row[f'pred_h{h+1}'] = float(y_pred[j, h])
        row[f'act_h{h+1}'] = float(y_test_true[j, h])
    pred_rows.append(row)

pred_df = pd.DataFrame(pred_rows).set_index('timestamp')
pred_df.to_csv(os.path.join(OUT_DIR, 'forecast.csv'))

print('Saved forecasts to', os.path.join(OUT_DIR, 'forecast_fixed.csv'))

# plot first horizon comparison
try:
    plt.figure(figsize=(12,5))
    plt.plot(pred_df.index[:200], pred_df['act_h1'].values[:200], label='actual')
    plt.plot(pred_df.index[:200], pred_df['pred_h1'].values[:200], label='predicted')
    plt.title('CNN-LSTM Forecast (h1) vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'forecast_vs_actual_h1_fixed.png'))
    plt.close()
except Exception:
    pass

# Save model & scalers
if joblib:
    model.save(os.path.join(MODEL_DIR, 'cnn_lstm_fixed.h5'))
    joblib.dump(sc_X, os.path.join(MODEL_DIR, 'sc_X.pkl'))
    joblib.dump(sc_y, os.path.join(MODEL_DIR, 'sc_y.pkl'))
    joblib.dump(model, os.path.join(MODEL_DIR, 'cnn_lstm_obj.pkl'))

# -------------------------
# Additional diagnostics (ACF/PACF)
# -------------------------
if plot_acf is not None:
    try:
        plt.figure(figsize=(10,4))
        plot_acf(df[target].dropna().values[:1440], lags=50)
        plt.title('Autocorrelation (first 10 days)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'acf.png'))
        plt.close()
    except Exception:
        pass
else:
    print('statsmodels not available; skipping ACF/PACF plots')

print('All done. Outputs saved to', OUT_DIR)
