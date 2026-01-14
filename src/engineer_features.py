import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import calculate_emas, process_greeks, calculate_derived_features, aggregate_option_metrics
from src.data_utils import clean_data

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    # Load clean data
    print("Loading clean data...")
    spot_df = pd.read_csv(os.path.join(data_dir, 'nifty_spot_5min_clean.csv'))
    futures_df = pd.read_csv(os.path.join(data_dir, 'nifty_futures_5min_clean.csv'))
    options_df = pd.read_csv(os.path.join(data_dir, 'nifty_options_5min_clean.csv'))
    
    # Ensure timestamps
    spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
    futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
    options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
    
    # Rename spot columns for clarity (to avoid confusion after merge)
    spot_df = spot_df.rename(columns={'close': 'close_spot', 'open': 'open_spot', 
                                       'high': 'high_spot', 'low': 'low_spot', 
                                       'volume': 'volume_spot'})
    
    # 1. EMAs on Spot
    spot_df = calculate_emas(spot_df)
    
    # 2. Aggregated Option Metrics (PCR, etc.)
    agg_options = aggregate_option_metrics(options_df)
    agg_options['timestamp'] = pd.to_datetime(agg_options['timestamp'])
    
    # 3. ATM Greeks
    # We need to identify the ATM option for each timestamp.
    # In our synthetic generation, we have strikes. 
    # Let's filter options_df for the rows that are closest to ATM.
    # Since we don't have a "is_atm" flag, we derive it.
    # Actually, simpler: compute greeks for ALL options, then filter for ATM.
    
    # First, merge options with spot to have 'close_spot' for Greeks calculation
    options_with_spot = pd.merge(options_df, spot_df[['timestamp', 'close_spot']], on='timestamp')
    
    # Compute Greeks for all options (might be slow but robust)
    print("Computing Greeks for options...")
    options_with_greeks = process_greeks(options_with_spot)
    
    # Now select ATM Call and Put (already has close_spot)
    opt_merge = options_with_greeks
    
    # Define ATM: min abs(strike - close)
    # We can do this efficiently by sorting and taking head(1) per timestamp/type?
    # Or just picking the one where diff is min.
    
    opt_merge['dist'] = abs(opt_merge['strike'] - opt_merge['close_spot'])
    
    # Select ATM Call
    atm_calls = opt_merge[opt_merge['type'] == 'CE'].sort_values('dist').groupby('timestamp').first().reset_index()
    # Rename columns to be specific
    atm_calls = atm_calls[['timestamp', 'delta', 'gamma', 'theta', 'vega', 'rho', 'oi']]
    atm_calls.columns = ['timestamp', 'call_delta', 'call_gamma', 'call_theta', 'call_vega', 'call_rho', 'call_atm_oi']
    
    # Select ATM Put
    atm_puts = opt_merge[opt_merge['type'] == 'PE'].sort_values('dist').groupby('timestamp').first().reset_index()
    atm_puts = atm_puts[['timestamp', 'delta', 'gamma', 'theta', 'vega', 'rho', 'oi']]
    atm_puts.columns = ['timestamp', 'put_delta', 'put_gamma', 'put_theta', 'put_vega', 'put_rho', 'put_atm_oi']
    
    # 4. Merge everything into Final Dataset
    # Base: Spot
    final_df = spot_df.copy()
    
    # Merge Futures (Close, OI)
    final_df = pd.merge(final_df, futures_df[['timestamp', 'close', 'oi']].rename(columns={'close': 'close_fut'}), on='timestamp')
    
    # Merge Aggregated Options
    final_df = pd.merge(final_df, agg_options, on='timestamp', how='left')
    
    # Merge ATM Greeks
    final_df = pd.merge(final_df, atm_calls, on='timestamp', how='left')
    final_df = pd.merge(final_df, atm_puts, on='timestamp', how='left')
    
    # 5. Derived Features
    # Now we have columns like 'call_iv' (from agg mainly? No, agg gave avg_iv).
    # Task 2.3 asks for "Delta Neutral Ratio = abs(call delta) / abs(put delta)"
    # We have call_delta and put_delta now.
    
    final_df = calculate_derived_features(final_df) # This handles Basis, Returns
    
    # Additional Derived from Task 2.3
    final_df['delta_neutral_ratio'] = abs(final_df['call_delta']) / abs(final_df['put_delta'])
    
    # Gamma Exposure = spot close * gamma * open interest
    # We will use ATM Gamma and ATM OI (Total of Call+Put or just one? Usually summed for market exposure, but let's take ATM Call Gamma * Call OI + Put Gamma * Put OI?)
    # Task 2.3 just says "Gamma Exposure = spot close * gamma * open interest".
    # We'll calculate it as Spot * (CallGamma*CallOI + PutGamma*PutOI) to represent total exposure at ATM.
    
    if 'call_gamma' in final_df.columns:
        final_df['gamma_exposure'] = final_df['close_spot'] * (
            (final_df['call_gamma'] * final_df['call_atm_oi']) + 
            (final_df['put_gamma'] * final_df['put_atm_oi'])
        )
    
    # Save
    out_path = os.path.join(data_dir, 'nifty_features_5min.csv')
    final_df.to_csv(out_path, index=False)
    print(f"Saved Final Feature Set to {out_path} ({len(final_df)} rows)")

if __name__ == "__main__":
    main()
