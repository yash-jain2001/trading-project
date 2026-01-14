import pandas as pd
import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("matplotlib/seaborn not installed")

class OutlierAnalyzer:
    """
    Identifies and analyzes profitable trades beyond 3-sigma (Z-score > 3).
    """
    def __init__(self, trades_df, features_df):
        self.trades_df = trades_df.copy()
        self.features_df = features_df.copy()
        self.outlier_trades = None
        self.normal_trades = None
        
    def identify_outliers(self, z_threshold=3):
        """
        Identify profitable trades with Z-score > 3.
        """
        profitable = self.trades_df[self.trades_df['pnl'] > 0].copy()
        
        if len(profitable) == 0:
            print("No profitable trades found")
            return
            
        mean_pnl = profitable['pnl'].mean()
        std_pnl = profitable['pnl'].std()
        
        if std_pnl == 0:
            print("Standard deviation is 0 - cannot compute Z-scores")
            return
            
        profitable['z_score'] = (profitable['pnl'] - mean_pnl) / std_pnl
        
        self.outlier_trades = profitable[profitable['z_score'] > z_threshold]
        self.normal_trades = profitable[profitable['z_score'] <= z_threshold]
        
        print(f"Total Profitable Trades: {len(profitable)}")
        print(f"Outlier Trades (Z > {z_threshold}): {len(self.outlier_trades)}")
        print(f"Normal Profitable Trades: {len(self.normal_trades)}")
        
        return self.outlier_trades
        
    def get_trade_features(self, trades):
        """
        Get features at entry time for given trades.
        """
        if trades is None or trades.empty:
            return None
            
        trades = trades.copy()
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        self.features_df['timestamp'] = pd.to_datetime(self.features_df['timestamp'])
        
        merged = pd.merge_asof(
            trades.sort_values('entry_time'),
            self.features_df.sort_values('timestamp'),
            left_on='entry_time',
            right_on='timestamp',
            direction='backward'
        )
        return merged
        
    def compare_statistics(self):
        """
        Compare outlier trades vs normal profitable trades.
        """
        if self.outlier_trades is None or self.normal_trades is None:
            print("Run identify_outliers first")
            return
            
        outlier_features = self.get_trade_features(self.outlier_trades)
        normal_features = self.get_trade_features(self.normal_trades)
        
        if outlier_features is None or normal_features is None:
            return
            
        # Compute statistics
        stats = []
        
        numeric_cols = [c for c in outlier_features.columns if outlier_features[c].dtype in ['float64', 'int64']]
        
        for col in numeric_cols:
            if col in ['pnl', 'z_score', 'entry', 'exit']:
                continue
            try:
                outlier_mean = outlier_features[col].mean()
                normal_mean = normal_features[col].mean()
                diff_pct = ((outlier_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
                
                stats.append({
                    'feature': col,
                    'outlier_mean': outlier_mean,
                    'normal_mean': normal_mean,
                    'diff_pct': diff_pct
                })
            except:
                pass
                
        return pd.DataFrame(stats).sort_values('diff_pct', key=abs, ascending=False)
        
    def generate_summary(self):
        """
        Generate insights summary.
        """
        if self.outlier_trades is None:
            self.identify_outliers()
            
        total_trades = len(self.trades_df)
        profitable_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        outlier_count = len(self.outlier_trades) if self.outlier_trades is not None else 0
        
        summary = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'outlier_count': outlier_count,
            'outlier_pct': (outlier_count / profitable_trades * 100) if profitable_trades > 0 else 0,
            'avg_outlier_pnl': self.outlier_trades['pnl'].mean() if outlier_count > 0 else 0,
            'avg_normal_pnl': self.normal_trades['pnl'].mean() if self.normal_trades is not None and len(self.normal_trades) > 0 else 0
        }
        
        return summary

def create_visualizations(trades_df, features_df, plots_dir):
    """
    Create visualization plots for the analysis.
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting not available")
        return
        
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. PnL Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Break Even')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.title('Trade PnL Distribution')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'pnl_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. PnL vs Duration (if duration available)
    if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 60
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        plt.scatter(trades_df['duration'], trades_df['pnl'], c=colors, alpha=0.6)
        plt.xlabel('Duration (minutes)')
        plt.ylabel('PnL')
        plt.title('PnL vs Trade Duration')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.savefig(os.path.join(plots_dir, 'pnl_vs_duration.png'), dpi=100, bbox_inches='tight')
        plt.close()
    
    # 3. Price chart with regimes (if regime column exists)
    if 'regime' in features_df.columns and 'close_spot' in features_df.columns:
        # Sample for performance
        sample_df = features_df.iloc[::10].copy()  # Every 10th row
        
        plt.figure(figsize=(14, 6))
        plt.plot(sample_df['timestamp'], sample_df['close_spot'], color='black', linewidth=0.5)
        
        # Color by regime
        for regime, color in [(1, 'green'), (-1, 'red'), (0, 'yellow')]:
            mask = sample_df['regime'] == regime
            plt.scatter(sample_df.loc[mask, 'timestamp'], sample_df.loc[mask, 'close_spot'], 
                       c=color, s=1, alpha=0.5, label=f'Regime {regime}')
        
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Price with Regime Overlay')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'price_regime.png'), dpi=100, bbox_inches='tight')
        plt.close()
        
    print(f"Saved visualizations to {plots_dir}")

def run_analysis():
    """
    Main analysis execution.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    plots_dir = os.path.join(os.path.dirname(__file__), '../plots')
    
    # Load data
    features_path = os.path.join(data_dir, 'nifty_features_5min.csv')
    trades_path = os.path.join(results_dir, 'backtest_trades.csv')
    
    if not os.path.exists(trades_path):
        print("Trades file not found. Run backtest first.")
        return
        
    features_df = pd.read_csv(features_path)
    trades_df = pd.read_csv(trades_path)
    
    print(f"Loaded {len(trades_df)} trades for analysis")
    
    # Outlier Analysis
    analyzer = OutlierAnalyzer(trades_df, features_df)
    outliers = analyzer.identify_outliers(z_threshold=3)
    
    summary = analyzer.generate_summary()
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
        
    # Feature comparison
    stats = analyzer.compare_statistics()
    if stats is not None and not stats.empty:
        print("\n=== Top Distinguishing Features ===")
        print(stats.head(10).to_string())
        stats.to_csv(os.path.join(results_dir, 'outlier_feature_comparison.csv'), index=False)
    
    # Visualizations
    create_visualizations(trades_df, features_df, plots_dir)
    
    # Save summary
    with open(os.path.join(results_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("=== Outlier Trade Analysis Summary ===\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
            
    print(f"\nAnalysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    run_analysis()
