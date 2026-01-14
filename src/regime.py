import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    pass # Will handle import error or mock if not installed

class RegimeDetector:
    def __init__(self, n_states=3):
        self.n_states = n_states
        # GaussianHMM might fail if not installed
        try:
            self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
            self.installed = True
        except:
            print("hmmlearn not installed. Regime detection will be mocked.")
            self.installed = False
            self.model = None

    def fit(self, X):
        """
        Fits the HMM model on the data.
        X: 2D array of features
        """
        if not self.installed:
            return
        self.model.fit(X)

    def predict(self, X):
        """
        Predicts the state for each sample.
        Returns: array of states
        """
        if not self.installed:
            # Mock regimes: 0=Sideways, 1=Uptrend, -1=Downtrend (mapped later)
            # Return random states 0, 1, 2
            return np.random.randint(0, self.n_states, size=len(X))
        
        hidden_states = self.model.predict(X)
        return hidden_states

    def map_regimes(self, df, state_col='hidden_state', price_col='close_spot'):
        """
        Maps hidden states (0,1,2) to meaningful labels (+1, -1, 0)
        based on average returns of that state.
        +1: Uptrend (Highest positive return)
        -1: Downtrend (Lowest/Negative return)
         0: Sideways
        """
        if 'spot_returns' not in df.columns:
            df['spot_returns'] = df[price_col].pct_change().fillna(0)
            
        summary = df.groupby(state_col)['spot_returns'].mean()
        print("State Returns Summary:\n", summary)
        
        sorted_states = summary.sort_values(ascending=False).index.tolist()
        # sorted[0] -> Highest return (Uptrend +1)
        # sorted[-1] -> Lowest/Negative return (Downtrend -1)
        # sorted[1...n-1] -> Sideways (0)
        
        mapping = {}
        if len(sorted_states) >= 3:
            mapping[sorted_states[0]] = 1   # Uptrend
            mapping[sorted_states[-1]] = -1 # Downtrend
            for s in sorted_states[1:-1]:
                mapping[s] = 0              # Sideways
        else:
            # Fallback for <3 states
            mapping = {s: 0 for s in sorted_states}
            mapping[sorted_states[0]] = 1
            if len(sorted_states) > 1:
                mapping[sorted_states[-1]] = -1
                
        print("Regime Mapping:", mapping)
        return df[state_col].map(mapping)

def detect_regimes(df, features_list):
    """
    Main function to detect regimes.
    Train on first 70%?? Task 3.1 says "Train on first 70% of data".
    But we need regimes for the whole dataset for strategy backtesting (or just test set?).
    Usually training on past, predicting on future.
    For this assignment, we fit on Train(70%) and predict on Test(30%)? 
    Or fit on Train, predict on All? 
    "Implement HMM... Train on first 70% of data."
    Task 4.1: "Implement 5/15 EMA strategy with regime filter... Backtest... Last 30%".
    So we need predicted regimes for the last 30%.
    
    Approach:
    1. Split data.
    2. Fit HMM on Train.
    3. Predict hidden states for All (or just persist state? HMM matches patterns).
       Ideally, we predict on Train, and then for Test we predict state based on observations.
    """
    print("Detecting Regimes...")
    
    # Drop NA in features
    data = df[features_list].dropna()
    # Align indices
    df_aligned = df.loc[data.index].copy()
    
    split_idx = int(len(data) * 0.70)
    train_data = data.iloc[:split_idx]
    
    detector = RegimeDetector(n_states=3)
    detector.fit(train_data.values)
    
    # Predict for ALL data
    all_states = detector.predict(data.values)
    
    df_aligned['hidden_state'] = all_states
    
    # Map regimes using Training data statistics only? Or All?
    # To avoid lookahead bias, we should map based on Train stats.
    
    # Get stats from Train part
    df_train_with_states = df_aligned.iloc[:split_idx]
    regime_map = {}
    
    # Re-implement mapping helper inside here or use the class measure
    # But detector doesn't store returns.
    
    if 'spot_returns' not in df_aligned.columns:
        df_aligned['spot_returns'] = df_aligned['close_spot'].pct_change().fillna(0)
        
    train_check = df_aligned.iloc[:split_idx]
    summary = train_check.groupby('hidden_state')['spot_returns'].mean()
    sorted_states = summary.sort_values(ascending=False).index.tolist()
    
    mapping = {}
    if len(sorted_states) >= 3:
        mapping[sorted_states[0]] = 1   # Uptrend
        mapping[sorted_states[-1]] = -1 # Downtrend
        for s in sorted_states[1:-1]:
            mapping[s] = 0              # Sideways
    else:
        mapping = {s: 0 for s in sorted_states}
        
    df_aligned['regime'] = df_aligned['hidden_state'].map(mapping)
    
    # Merge back to original df
    df['regime'] = np.nan
    df.loc[df_aligned.index, 'regime'] = df_aligned['regime']
    
    # Forward fill regime for missing rows (if any) or fill 0
    df['regime'] = df['regime'].fillna(0)
    
    return df

if __name__ == "__main__":
    pass
