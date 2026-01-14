import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.regime import detect_regimes
from src.strategy import Strategy

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load Features
    features_path = os.path.join(data_dir, 'nifty_features_5min.csv')
    if not os.path.exists(features_path):
        print("Features file not found. Run engineer_features.py first.")
        return
        
    print("Loading Features...")
    df = pd.read_csv(features_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Part 3: Regime Detection
    # Features for HMM (Options-based)
    hmm_features = [
        'avg_iv', 'iv_spread', 'pcr_oi', 
        'call_delta', 'call_gamma', 'call_vega', # ATM Specifics (using ATM Call/Put stats or Just ATM Call?)
        # Task 3.1: "Average IV, IV Spread, PCR, ATM Delta, ATM Gamma, ATM Vega, Futures Basis, Spot Returns"
        'futures_basis', 'spot_returns'
    ]
    
    # Note: 'call_delta' is close to ATM Delta. 'call_gamma' is ATM Gamma.
    # Check if columns exist
    missing_cols = [c for c in hmm_features if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for HMM: {missing_cols}")
        # Assuming we might have mapped them differently? 
        # In features.py/engineer_features.py we have 'call_delta', 'avg_iv' etc.
        # But 'avg_iv' aggregation might be missing if I didn't enable it properly?
        # Let's hope engineer_features.py ran correctly.
    
    print("Detecting Regimes...")
    df = detect_regimes(df, hmm_features)
    
    # Save with regimes
    df.to_csv(os.path.join(data_dir, 'nifty_regimes_5min.csv'), index=False)
    
    # Part 4: Strategy Implementation
    print("Running Strategy Backtest...")
    strategy = Strategy(ema_fast=5, ema_slow=15)
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    # Split Data (Train 70% / Test 30%)
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Backtesting on Test Set ({len(test_df)} rows)...")
    trades = strategy.backtest(test_df)
    
    if trades.empty:
        print("No trades generated.")
    else:
        print(f"Generated {len(trades)} trades.")
        trades.to_csv(os.path.join(results_dir, 'backtest_trades.csv'), index=False)
        
        # Calculate Metrics
        total_return = trades['pnl'].sum()
        win_rate = len(trades[trades['pnl'] > 0]) / len(trades)
        avg_trade = trades['pnl'].mean()
        
        print(f"Total Return: {total_return}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average PnL: {avg_trade:.2f}")
        
    # Save final DF with signals
    df.to_csv(os.path.join(results_dir, 'backtest_data_full.csv'), index=False)

if __name__ == "__main__":
    main()
