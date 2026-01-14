import pandas as pd
import numpy as np
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed")

# LSTM would require tensorflow/keras - making it optional
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed - LSTM will be skipped")

class TradePredictor:
    """
    Binary classifier to predict if a trade signal will be profitable.
    """
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.feature_cols = None
        
    def prepare_features(self, df, trades_df):
        """
        Prepare features for ML training.
        For each trade, we need features at the time of signal.
        Target: 1 if trade profitable, 0 otherwise.
        """
        if trades_df.empty:
            return None, None
            
        # Add profitability target
        trades_df = trades_df.copy()
        trades_df['target'] = (trades_df['pnl'] > 0).astype(int)
        
        # For each trade, get features at entry time
        # This is a simplified approach - ideally match exact timestamp
        
        # Merge trades with features at entry time
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get closest feature row for each trade
        merged = pd.merge_asof(
            trades_df.sort_values('entry_time'),
            df.sort_values('timestamp'),
            left_on='entry_time',
            right_on='timestamp',
            direction='backward'
        )
        
        # Select feature columns (exclude target, timestamps, etc.)
        exclude_cols = ['timestamp', 'entry_time', 'exit_time', 'target', 'pnl', 
                        'entry', 'exit', 'type', 'signal', 'entry_signal', 'exit_signal',
                        'prev_ema_5', 'prev_ema_15']
        
        feature_cols = [c for c in merged.columns if c not in exclude_cols and merged[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        self.feature_cols = feature_cols
        
        X = merged[feature_cols].fillna(0)
        y = merged['target']
        
        return X, y
    
    def train_xgboost(self, X, y, test_size=0.3):
        """
        Train XGBoost classifier with time-series cross-validation.
        """
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            print("XGBoost or sklearn not available")
            return None
            
        print("Training XGBoost model...")
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.xgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"XGBoost Test Accuracy: {acc:.2%}")
        
        return self.xgb_model
        
    def train_lstm(self, X, y, sequence_length=10, test_size=0.3):
        """
        Train LSTM model for sequence-based prediction.
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - skipping LSTM")
            return None
            
        print("Training LSTM model...")
        
        # Create sequences
        X_seq, y_seq = [], []
        X_vals = X.values
        y_vals = y.values
        
        for i in range(sequence_length, len(X_vals)):
            X_seq.append(X_vals[i-sequence_length:i])
            y_seq.append(y_vals[i])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        loss, acc = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM Test Accuracy: {acc:.2%}")
        
        return self.lstm_model
        
    def predict_xgb(self, X):
        """Get XGBoost predictions and probabilities."""
        if self.xgb_model is None:
            return None, None
        preds = self.xgb_model.predict(X)
        probs = self.xgb_model.predict_proba(X)[:, 1]
        return preds, probs
        
    def save_models(self, model_dir):
        """Save trained models."""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.xgb_model is not None:
            import joblib
            joblib.dump(self.xgb_model, os.path.join(model_dir, 'xgb_model.pkl'))
            print(f"Saved XGBoost model to {model_dir}")
            
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(model_dir, 'lstm_model.h5'))
            print(f"Saved LSTM model to {model_dir}")

def run_ml_pipeline():
    """
    Main ML pipeline execution.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    
    # Load data
    features_path = os.path.join(data_dir, 'nifty_features_5min.csv')
    trades_path = os.path.join(results_dir, 'backtest_trades.csv')
    
    if not os.path.exists(features_path) or not os.path.exists(trades_path):
        print("Required files not found. Run backtest first.")
        return
        
    df = pd.read_csv(features_path)
    trades_df = pd.read_csv(trades_path)
    
    print(f"Loaded {len(df)} feature rows and {len(trades_df)} trades")
    
    predictor = TradePredictor()
    
    X, y = predictor.prepare_features(df, trades_df)
    
    if X is None:
        print("No data to train on")
        return
        
    print(f"Prepared {len(X)} samples with {len(predictor.feature_cols)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train models
    predictor.train_xgboost(X, y)
    predictor.train_lstm(X, y)
    
    # Save models
    predictor.save_models(models_dir)

if __name__ == "__main__":
    run_ml_pipeline()
