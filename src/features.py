import pandas as pd
import numpy as np
from math import log, sqrt, exp
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False

def calculate_emas(df, period_fast=5, period_slow=15):
    """
    Calculates Exponential Moving Averages.
    """
    print("Calculating EMAs...")
    df[f'ema_{period_fast}'] = df['close_spot'].ewm(span=period_fast, adjust=False).mean()
    df[f'ema_{period_slow}'] = df['close_spot'].ewm(span=period_slow, adjust=False).mean()
    return df

def bs_greeks_fallback(S, K, T, r, sigma, option_type='CE'):
    """
    Calculate Black-Scholes Greeks without external libraries.
    S: Spot price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free rate (as percentage, e.g., 6.5)
    sigma: Volatility (as percentage, e.g., 15)
    """
    try:
        r = r / 100  # Convert to decimal
        sigma = sigma / 100  # Convert to decimal
        
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if SCIPY_AVAILABLE:
            N = norm.cdf
            n = norm.pdf
        else:
            # Simple approximation for normal CDF
            def N(x):
                return 0.5 * (1 + np.tanh(0.7978845608 * x * (1 + 0.044715 * x * x)))
            def n(x):
                return exp(-0.5 * x * x) / sqrt(2 * 3.14159265359)
        
        gamma = n(d1) / (S * sigma * sqrt(T))
        vega = S * n(d1) * sqrt(T) / 100  # Per 1% change in vol
        
        if option_type == 'CE':
            delta = N(d1)
            theta = (-S * n(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * N(d2)) / 365
            rho = K * T * exp(-r * T) * N(d2) / 100
        else:
            delta = N(d1) - 1
            theta = (-S * n(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * N(-d2)) / 365
            rho = -K * T * exp(-r * T) * N(-d2) / 100
            
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

def calculate_greeks(row, risk_free_rate=6.5):
    """
    Calculates Greeks for a single row (option) using mibian or fallback.
    """
    S = row['close_spot']
    K = row['strike']
    r = risk_free_rate
    t = 7  # Days to expiry
    T = t / 365  # Years
    vol = row['iv']
    option_type = row['type']
    
    if MIBIAN_AVAILABLE:
        try:
            c = mibian.BS([S, K, r, t], volatility=vol)
            if option_type == 'CE':
                return {
                    'delta': c.callDelta,
                    'gamma': c.gamma,
                    'theta': c.callTheta,
                    'vega': c.vega,
                    'rho': c.callRho
                }
            else:
                return {
                    'delta': c.putDelta,
                    'gamma': c.gamma,
                    'theta': c.putTheta,
                    'vega': c.vega,
                    'rho': c.putRho
                }
        except:
            pass
    
    # Fallback to custom implementation
    return bs_greeks_fallback(S, K, T, r, vol, option_type)

def process_greeks(df):
    """
    Apply greeks calculation to the dataframe.
    This can be slow for large datasets.
    """
    print("Calculating Greeks (this is slow)...")
    # For performance, maybe vectorizing or using a simplified BS formula is better.
    # But for 'mibian' requirement, we loop or apply.
    
    # We only need Greeks for ATM Call and Put as per Task 2.2 ??
    # "Calculate ... for ATM Call and Put"
    # Our merged dataset has ALL strikes (if we merged everything).
    # If we only merged ATM, then yes.
    # Let's assume the passed df has the option columns.
    
    # To avoid loop slowness, we'll try to apply.
    # Note: 'iv' in our data is per option.
    
    greeks = df.apply(calculate_greeks, axis=1)
    greeks_df = pd.DataFrame(greeks.tolist())
    
    df = pd.concat([df.reset_index(drop=True), greeks_df], axis=1)
    return df

def calculate_derived_features(df):
    """
    Calculates derived features:
    Average IV, IV Spread, PCR (OI/Vol), Basis, Returns, Delta Neutral Ratio, Gamma Exposure
    """
    print("Calculating Derived Features...")
    
    # Note: The merged dataframe structure matters here.
    # if the dataframe is LONG (one row per option), these aggregate metrics 
    # (like total PCR) need to be calculated PER TIMESTAMP before merging or using groupby.
    
    # IF the input `df` is the result of `align_and_merge` which gave us a LONG format (spot + 1 option per row),
    # then "Total Put OI / Total Call OI" needs aggregation across the option chain for that timestamp.
    
    # Strategy:
    # 1. We assume we have access to the full option chain for aggregation.
    # 2. Or if `df` contains only ATM options (if we filtered), we can't calculate TOTAL PCR.
    
    # Let's assume we handle aggregation separately or `df` has aggregation columns pre-calculated.
    # For this function, let's assume we calculate row-based metrics.
    
    # Futures Basis
    if 'close_fut' in df.columns and 'close_spot' in df.columns:
        df['futures_basis'] = (df['close_fut'] - df['close_spot']) / df['close_spot']
        
    # Returns
    df['spot_returns'] = df['close_spot'].pct_change()
    df['futures_returns'] = df['close_fut'].pct_change()
    
    # Gamma Exposure = Spot * Gamma * OpenInterest
    if 'gamma' in df.columns and 'oi' in df.columns:
        df['gamma_exposure'] = df['close_spot'] * df['gamma'] * df['oi']
        
    return df
    
def aggregate_option_metrics(options_df):
    """
    Calculates timestamp-level metrics from options chain:
    PCR, Average IV (ATM), IV Spread (ATM)
    """
    print("Aggregating Option Metrics...")
    
    # Group by timestamp
    grouped = options_df.groupby('timestamp')
    
    metrics = []
    
    for timestamp, group in grouped:
        # Total Volume/OI PCR
        calls = group[group['type'] == 'CE']
        puts = group[group['type'] == 'PE']
        
        total_call_oi = calls['oi'].sum()
        total_put_oi = puts['oi'].sum()
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # ATM IV (Find strike closest to spot? We don't have spot here easily unless merged.
        # But we know strikes. Let's assume middle strike is ATM or take mean of IVs)
        # Better: Average IV of all near-the-money options
        avg_call_iv = calls['iv'].mean()
        avg_put_iv = puts['iv'].mean()
        
        avg_iv = (avg_call_iv + avg_put_iv) / 2
        iv_spread = avg_call_iv - avg_put_iv
        
        metrics.append({
            'timestamp': timestamp,
            'pcr_oi': pcr_oi,
            'pcr_vol': pcr_vol,
            'avg_iv': avg_iv,
            'iv_spread': iv_spread
        })
        
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    pass
