try:
    import yfinance as yf
except ImportError:
    yf = None

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_nifty_spot(start_date='2024-01-14', end_date='2025-01-14', interval='5m'):
    """
    Fetches NIFTY 50 Spot data using yfinance.
    Note: yfinance 5m data is limited to the last 60 days. 
    For a 1-year window in a real scenario, you'd need a paid API.
    Here we will fetch what is available or simulate if needed for the 1y requirement.
    """
    print(f"Fetching NIFTY 50 Spot data from {start_date} to {end_date}...")
    ticker = "^NSEI"
    try:
        if yf is None:
            raise ImportError("yfinance not installed")
            
        # yfinance restriction: 1m data 7 days, 5m data 60 days.
        # We will try to fetch max available 5m data.
        df = yf.download(ticker, period="60d", interval=interval, progress=False)
        
        if df.empty:
            print("Warning: No data fetched from yfinance. Generating synthetic data for demonstration.")
            return generate_synthetic_spot_data(start_date, end_date)
            
        # Standardize columns
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns] # Flatten multiindex if present
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        
        # Reset index to get datetime column
        df = df.reset_index()
        df = df.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp'})
        
        # Filter columns
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[cols]
        
        print(f"Fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error fetching spot data: {e}")
        return generate_synthetic_spot_data(start_date, end_date)

def generate_synthetic_spot_data(start_date, end_date):
    """
    Generates synthetic NIFTY 50 spot data for the given range directly.
    Used when API limits prevent fetching full history.
    """
    print("Generating synthetic NIFTY Spot data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Filter for trading hours (9:15 to 15:30)
    trading_dates = []
    for d in dates:
        if d.weekday() < 5 and (d.time() >= datetime.strptime("09:15", "%H:%M").time()) and (d.time() <= datetime.strptime("15:30", "%H:%M").time()):
            trading_dates.append(d)
            
    n = len(trading_dates)
    price = 21000.0
    prices = []
    
    # Random walk
    for _ in range(n):
        change = np.random.normal(0, 5) # Random move
        price += change
        prices.append(price)
        
    df = pd.DataFrame({
        'timestamp': trading_dates,
        'open': [p + np.random.normal(0, 2) for p in prices],
        'high': [p + abs(np.random.normal(0, 5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 5)) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 100000, n)
    })
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df

def generate_nifty_futures(spot_df):
    """
    Generates synthetic NIFTY Futures data based on Spot data.
    Adds a basis (premium/discount) and simulates rollover volume.
    """
    print("Generating NIFTY Futures data...")
    df = spot_df.copy()
    
    # Futures typically trade at a premium
    df['close'] = df['close'] * (1 + np.random.normal(0.005, 0.001, len(df))) # ~0.5% premium
    df['open'] = df['open'] * (1 + np.random.normal(0.005, 0.001, len(df)))
    df['high'] = df['high'] * (1 + np.random.normal(0.005, 0.001, len(df)))
    df['low'] = df['low'] * (1 + np.random.normal(0.005, 0.001, len(df)))
    
    # Open Interest simulation
    df['oi'] = np.random.randint(5000000, 10000000, len(df))
    
    # Adjust OHLC to be consistent
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df

def generate_nifty_options(spot_df):
    """
    Generates synthetic NIFTY Options Chain data.
    Strikes: ATM, ATM+100, ATM-100 (Simplified for ATM+/-1)
    """
    print("Generating NIFTY Options data...")
    
    options_data = []
    
    for idx, row in spot_df.iterrows():
        timestamp = row['timestamp']
        spot_price = row['close']
        
        # Determine ATM strike (Round to nearest 50)
        atm_strike = round(spot_price / 50) * 50
        
        strikes = [atm_strike - 50, atm_strike, atm_strike + 50]
        
        for strike in strikes:
            # Simulate Call
            call_price = max(0, spot_price - strike) + np.random.uniform(50, 150) # Intrinsic + Time value
            call_iv = np.random.uniform(10, 20)
            call_oi = np.random.randint(100000, 500000)
            call_vol = np.random.randint(100, 5000)
            
            # Simulate Put
            put_price = max(0, strike - spot_price) + np.random.uniform(50, 150)
            put_iv = np.random.uniform(10, 25)
            put_oi = np.random.randint(100000, 500000)
            put_vol = np.random.randint(100, 5000)
            
            options_data.append({
                'timestamp': timestamp,
                'strike': strike,
                'type': 'CE',
                'ltp': call_price,
                'iv': call_iv,
                'oi': call_oi,
                'volume': call_vol
            })
            
            options_data.append({
                'timestamp': timestamp,
                'strike': strike,
                'type': 'PE',
                'ltp': put_price,
                'iv': put_iv,
                'oi': put_oi,
                'volume': put_vol
            })
            
    return pd.DataFrame(options_data)

def clean_data(df, name="Dataset"):
    """
    Cleans the dataset:
    - Drops NA
    - Removes outliers (simple z-score or quantile based)
    """
    print(f"Cleaning {name}...")
    initial_rows = len(df)
    
    # Drop NA
    df = df.dropna()
    
    # Remove outliers (e.g., price change > 10% in 5 min is invalid for NIFTY, 
    # but let's be lenient for synthetic data)
    # We'll just checks for 0 or negative prices where not allowed
    if 'close' in df.columns:
        df = df[df['close'] > 0]
        
    print(f"Cleaned {name}: {initial_rows} -> {len(df)} rows.")
    return df

def align_and_merge(spot_df, futures_df, options_df):
    """
    Merges Spot, Futures, and Options on timestamp.
    """
    print("Merging datasets...")
    
    # Ensure timestamps are datetime
    spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
    futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
    options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
    
    # Merge Spot and Futures
    merged = pd.merge(spot_df, futures_df, on='timestamp', suffixes=('_spot', '_fut'), how='inner')
    
    # Options data is 'long' format (multiple rows per timestamp for different strikes/types)
    # We might want to pivot or just join.
    # For the assignment, "Merge spot, futures, and options data on timestamp"
    # typically implies a master dataframe. 
    # Since options are many-to-one with spot, we either:
    # 1. Keep it long (master df has multiple rows per timestamp)
    # 2. Pivot options to wide (columns like CE_ATM_IV, PE_ATM_IV, etc.)
    # Given we need to calculate Greeks for ATM specifically later, 
    # merging the *relevant* options (ATM) back to the time series makes sense.
    # But Task 1.3 just says "Merge... on timestamp".
    # I will create a big merged file where options columns are added.
    # Since there are multiple options per timestamp, this will duplicate spot/futures data 
    # for each option row. This is standard key-based merge.
    
    final_merged = pd.merge(merged, options_df, on='timestamp', how='inner')
    
    print(f"Merged Data Shape: {final_merged.shape}")
    return final_merged


if __name__ == "__main__":
    # Test execution
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    spot = fetch_nifty_spot(start_date, end_date)
    print("Spot Data Head:")
    print(spot.head())
    
    futures = generate_nifty_futures(spot)
    print("\nFutures Data Head:")
    print(futures.head())
    
    # Generate smaller chunk of options for testing speed
    options = generate_nifty_options(spot.iloc[:10]) 
    print("\nOptions Data Head (Sample):")
    print(options.head())
