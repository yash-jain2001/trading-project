import pandas as pd
import numpy as np
import os

def calculate_metrics(trades_df):
    """
    Calculate comprehensive backtest performance metrics.
    """
    if trades_df.empty:
        return {}
        
    trades = trades_df.copy()
    
    # Basic stats
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    losing_trades = len(trades[trades['pnl'] < 0])
    
    # Returns
    total_pnl = trades['pnl'].sum()
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    # Win rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Average trade
    avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    avg_trade = trades['pnl'].mean()
    
    # Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe Ratio (assuming daily returns, annualized)
    returns = trades['pnl']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Sortino Ratio (only downside deviation)
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    
    # Max Drawdown
    cumulative = returns.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = (total_pnl / abs(max_drawdown)) if max_drawdown != 0 else 0
    
    # Duration Analysis
    if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
        trades['duration'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 60
        avg_duration = trades['duration'].mean()
    else:
        avg_duration = 0
    
    metrics = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': f"{win_rate:.2%}",
        'Total Return': round(total_pnl, 2),
        'Gross Profit': round(gross_profit, 2),
        'Gross Loss': round(gross_loss, 2),
        'Avg Win': round(avg_win, 2),
        'Avg Loss': round(avg_loss, 2),
        'Avg Trade': round(avg_trade, 2),
        'Profit Factor': round(profit_factor, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Sortino Ratio': round(sortino, 2),
        'Max Drawdown': round(max_drawdown, 2),
        'Calmar Ratio': round(calmar, 2),
        'Avg Duration (min)': round(avg_duration, 2)
    }
    
    return metrics

def run_metrics():
    """
    Calculate and save metrics for backtest results.
    """
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    trades_path = os.path.join(results_dir, 'backtest_trades.csv')
    
    if not os.path.exists(trades_path):
        print("Trades file not found")
        return
        
    trades_df = pd.read_csv(trades_path)
    
    metrics = calculate_metrics(trades_df)
    
    print("\n=== Backtest Performance Metrics ===\n")
    for key, value in metrics.items():
        print(f"{key}: {value}")
        
    # Save to file
    with open(os.path.join(results_dir, 'performance_metrics.txt'), 'w') as f:
        f.write("=== Backtest Performance Metrics ===\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
            
    print(f"\nMetrics saved to {results_dir}/performance_metrics.txt")
    
    return metrics

if __name__ == "__main__":
    run_metrics()
