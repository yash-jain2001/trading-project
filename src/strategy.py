import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, ema_fast=5, ema_slow=15):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
    
    def generate_signals(self, df):
        """
        Generates trading signals based on 5/15 EMA crossover and Regime filter.
        df must have columns: 'ema_5', 'ema_15', 'regime'
        Regime: +1 (Uptrend), -1 (Downtrend), 0 (Sideways)
        """
        print("Generating Strategy Signals...")
        
        df = df.copy()
        df['signal'] = 0
        
        # Crossover Logic
        # We need previous candle values to detect crossover
        df['prev_ema_5'] = df['ema_5'].shift(1)
        df['prev_ema_15'] = df['ema_15'].shift(1)
        
        # Long Entry: 5 EMA crosses above 15 EMA AND Regime = +1
        # Cross above: Prev 5 < Prev 15 AND Curr 5 > Curr 15
        long_condition = (df['prev_ema_5'] < df['prev_ema_15']) & \
                         (df['ema_5'] > df['ema_15']) & \
                         (df['regime'] == 1)
                         
        # Short Entry: 5 EMA crosses below 15 EMA AND Regime = -1
        short_condition = (df['prev_ema_5'] > df['prev_ema_15']) & \
                          (df['ema_5'] < df['ema_15']) & \
                          (df['regime'] == -1)
                          
        # Long Exit: 5 EMA crosses below 15 EMA
        long_exit = (df['prev_ema_5'] > df['prev_ema_15']) & \
                    (df['ema_5'] < df['ema_15'])
                    
        # Short Exit: 5 EMA crosses above 15 EMA
        short_exit = (df['prev_ema_5'] < df['prev_ema_15']) & \
                     (df['ema_5'] > df['ema_15'])
                     
        # Vectorized signal generation (1=Buy, -1=Sell, 2=Exit Long, -2=Exit Short for easier processing?)
        # Or just 'position' state?
        # Let's iterate to manage state (more accurate for backtesting)
        # OR use vector logic for entries/exits and fill.
        
        # Let's use loop for clarity in backtesting logic usually, 
        # but for signal column generation:
        
        df.loc[long_condition, 'entry_signal'] = 1
        df.loc[short_condition, 'entry_signal'] = -1
        df.loc[long_exit, 'exit_signal'] = 1 # Exit Long
        df.loc[short_exit, 'exit_signal'] = -1 # Exit Short
        
        return df

    def backtest(self, df, initial_capital=100000):
        """
        Simulates the strategy.
        Assumes we enter/exit at Next Open after signal.
        """
        print("Backtesting Strategy...")
        # Ensure df is sorted by time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        position = 0 # 0=Flat, 1=Long, -1=Short
        entry_price = 0
        trades = []
        equity = [initial_capital]
        
        for i in range(1, len(df)-1):
            # Check for signals generated at candle i (using Close of i)
            # Trade Execute at Open of i+1
            
            current_row = df.iloc[i]
            next_row = df.iloc[i+1]
            
            # Logic variables
            ema_5 = current_row['ema_5']
            ema_15 = current_row['ema_15']
            prev_ema_5 = df.iloc[i-1]['ema_5']
            prev_ema_15 = df.iloc[i-1]['ema_15']
            regime = current_row['regime']
            
            # Crosses
            cross_up = (prev_ema_5 < prev_ema_15) and (ema_5 > ema_15)
            cross_down = (prev_ema_5 > prev_ema_15) and (ema_5 < ema_15)
            
            # Execution Price
            exec_price = next_row['open_spot'] if 'open_spot' in df.columns else next_row.get('open', next_row['close_spot'])
            
            # Exits first
            if position == 1 and cross_down:
                # Exit Long
                pnl = exec_price - entry_price
                trades.append({'type': 'Long', 'entry': entry_price, 'exit': exec_price, 'pnl': pnl, 'entry_time': entry_time, 'exit_time': next_row['timestamp']})
                position = 0
            
            elif position == -1 and cross_up:
                # Exit Short
                pnl = entry_price - exec_price
                trades.append({'type': 'Short', 'entry': entry_price, 'exit': exec_price, 'pnl': pnl, 'entry_time': entry_time, 'exit_time': next_row['timestamp']})
                position = 0
                
            # Entries
            if position == 0:
                if cross_up and regime == 1:
                    position = 1
                    entry_price = exec_price
                    entry_time = next_row['timestamp']
                    
                elif cross_down and regime == -1:
                    position = -1
                    entry_price = exec_price
                    entry_time = next_row['timestamp']
            
            # Mark to Market (simple)
            # equity.append(...)
            
        trades_df = pd.DataFrame(trades)
        return trades_df

if __name__ == "__main__":
    pass
