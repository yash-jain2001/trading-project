import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import fetch_nifty_spot, generate_nifty_futures, generate_nifty_options

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Fetch Spot
    # For full 1 year simulation, we might need synthetic data if yfinance is limited.
    # The utils handle this (yfinance 60d limit -> warnings/synthetic fallback).
    # Let's request 1 year and let the util decide or we force synthetic for consistency?
    # The util currently tries yfinance for 60d. If we want 1 year for the assignment, 
    # we should probably rely on the synthetic generator within the util for the FULL period 
    # to have a consistent 1-year dataset for backtesting.
    # However, the user asked for "Fetch 5-minute interval data for the last 1 year".
    # I will modify the call to ensure we get a good dataset.
    
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Step 1: Fetching/Generating Spot Data...")
    spot_df = fetch_nifty_spot(start_date, end_date)
    spot_path = os.path.join(data_dir, 'nifty_spot_5min.csv')
    spot_df.to_csv(spot_path, index=False)
    print(f"Saved Spot data to {spot_path}")
    
    # 2. Generate Futures
    print("Step 2: Generating Futures Data...")
    futures_df = generate_nifty_futures(spot_df)
    futures_path = os.path.join(data_dir, 'nifty_futures_5min.csv')
    futures_df.to_csv(futures_path, index=False)
    print(f"Saved Futures data to {futures_path}")
    
    # 3. Generate Options
    print("Step 3: Generating Options Data (this may take a moment)...")
    options_df = generate_nifty_options(spot_df)
    options_path = os.path.join(data_dir, 'nifty_options_5min.csv')
    options_df.to_csv(options_path, index=False)
    print(f"Saved Options data to {options_path}")

if __name__ == "__main__":
    main()
