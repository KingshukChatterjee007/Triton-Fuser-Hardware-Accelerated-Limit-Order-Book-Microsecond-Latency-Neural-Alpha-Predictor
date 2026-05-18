import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.python_bridge import FusedEngineBridge

def run_hft_backtester():
    print("="*80)
    print("TRITON FUSER: HIGH-FREQUENCY BACKTESTING & PORTFOLIO ENGINE")
    print("="*80)

    # 1. Setup paths and check files
    bin_file = "historical_itch_feed.bin"
    if not os.path.exists(bin_file):
        print(f"Empirical bin file {bin_file} not found. Creating a synthetic feed...")
        from scripts.generate_dummy_data import generate_dummy_data
        generate_dummy_data()
    
    # 2. Ingest L2 Ticks via AVX2 Ingestion Engine
    print("[Backtest Ingestion] Streaming raw order book ticks through C++ Bridge...")
    max_snapshots = 300
    n_bins = 100
    bin_width = 1  # Fixed: set bin_width=1 to align with simulated tick sizes!
    ticks_per_snapshot = 20

    bridge = FusedEngineBridge("src/ingestion_engine.dll")
    empirical_data = bridge.run_ingestion(
        bin_file, 
        n_bins=n_bins, 
        bin_width=bin_width, 
        max_snapshots=max_snapshots, 
        ticks_per_snapshot=ticks_per_snapshot
    )
    
    # empirical_data is [max_snapshots * n_bins, 3] containing (x_norm, t_norm, rho)
    total_snapshots = empirical_data.shape[0] // n_bins
    print(f"[Backtest Ingestion] Ingested {total_snapshots} full depth snapshots.")

    # 3. Simulate High-Frequency Trading Ledger
    print("[Backtest Execution] Initializing market simulation ledger...")
    
    # Setup HFT trading parameters
    initial_capital = 1000000.0  # $1,000,000 USD initial capital
    capital = initial_capital
    position = 0.0               # Current position in asset units
    cash = initial_capital
    
    # Costs & Latency Constraints
    fee_rate = 0.00005           # 0.5 bps (0.005%) VIP maker fee
    slippage_ticks = 0.1         # Slippage in tick units
    execution_delay = 1          # 1 snapshot execution delay (simulates network + engine latency)
    
    # Ledgers for analytics
    equity_curve = []
    benchmark_curve = []
    trade_logs = []
    
    # Reconstruct mid-prices from density centroids to act as the market price
    mid_prices = []
    fluid_velocities = []  # Rolling advection momentum indicator
    
    for i in range(total_snapshots):
        start_row = i * n_bins
        end_row = (i + 1) * n_bins
        snap_density = empirical_data[start_row:end_row, 2].numpy()
        
        # Mid-price proxy (center of mass of fluid density)
        centroid = np.dot(np.arange(n_bins), snap_density) / (np.sum(snap_density) + 1e-6)
        # Scale to match original price scale
        price = 100.0 + (centroid - 50.0) * bin_width
        mid_prices.append(price)
        
        # Calculate Rolling Advection Velocity (indicator u)
        if i >= 1:
            prev_start = (i - 1) * n_bins
            prev_end = i * n_bins
            prev_density = empirical_data[prev_start:prev_end, 2].numpy()
            prev_centroid = np.dot(np.arange(n_bins), prev_density) / (np.sum(prev_density) + 1e-6)
            
            u_momentum = centroid - prev_centroid
        else:
            u_momentum = 0.0
            
        fluid_velocities.append(u_momentum)

    print(f"[Backtest Execution] Running rolling trade simulation across {total_snapshots} snapshots...")

    # Calculate dynamic threshold based on volatility of fluid advection
    vel_std = np.std(fluid_velocities)
    threshold = 0.3 * vel_std if vel_std > 0 else 0.01
    print(f"[Backtest Execution] Calibrated dynamic indicator threshold: {threshold:.6f}")

    # We start trading from index 10 to allow indicators to warm up
    for t in range(10, total_snapshots - execution_delay):
        current_price = mid_prices[t]
        future_price_exec = mid_prices[t + execution_delay]  # Trade executes at future price due to delay!
        
        # Signal Generation from Advection Velocity (u)
        indicator = fluid_velocities[t]
        
        signal = 0
        if indicator > threshold:
            signal = 1   # BUY Signal
        elif indicator < -threshold:
            signal = -1  # SELL Signal
            
        # Execution logic with transaction fee and slippage
        executed = False
        execution_price = future_price_exec
        
        if signal == 1 and position <= 0:
            # Close short if exists
            if position < 0:
                cash -= abs(position) * execution_price * (1 + fee_rate)
                position = 0.0
                
            # Buy Asset (Go Long)
            execution_price = future_price_exec * (1 + fee_rate) + slippage_ticks
            allocated_cash = cash * 0.95  # Allocate 95% of cash to trade
            position_units = allocated_cash / execution_price
            
            cash_spent = position_units * execution_price
            cash -= cash_spent
            position = position_units
            executed = True
            trade_logs.append((t, 'BUY', execution_price, position_units, cash_spent * fee_rate))
            
        elif signal == -1 and position >= 0:
            # Sell Asset (Go Short or Liquidate Long)
            if position > 0:
                # Sell long position
                execution_price = future_price_exec * (1 - fee_rate) - slippage_ticks
                cash_received = position * execution_price
                cash += cash_received
                position = 0.0
                executed = True
                trade_logs.append((t, 'SELL_CLOSE', execution_price, position, cash_received * fee_rate))
                
            # Open Short Position (Borrow and Sell)
            execution_price = future_price_exec * (1 - fee_rate) - slippage_ticks
            allocated_capital = cash * 0.95
            position_units = -allocated_capital / execution_price
            
            cash_received = abs(position_units) * execution_price
            cash += cash_received
            position = position_units
            executed = True
            trade_logs.append((t, 'SHORT', execution_price, abs(position_units), cash_received * fee_rate))
            
        # Record Current Total Portfolio Value (Cash + Asset Value)
        portfolio_value = cash + position * current_price
        equity_curve.append(portfolio_value)
        
        # Benchmark Buy and Hold value
        benchmark_value = initial_capital * (current_price / mid_prices[10])
        benchmark_curve.append(benchmark_value)

    # 4. Calculate Portfolio Metrics & Statistics
    equity_curve = np.array(equity_curve)
    benchmark_curve = np.array(benchmark_curve)
    
    total_returns = (equity_curve[-1] - initial_capital) / initial_capital * 100
    benchmark_returns = (benchmark_curve[-1] - initial_capital) / initial_capital * 100
    
    # Daily equivalent log returns
    pct_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Annualized Sharpe Ratio (HFT scaling assumes high frequency activity)
    sharpe = np.mean(pct_returns) / (np.std(pct_returns) + 1e-9) * np.sqrt(252 * 1000)
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = np.min(drawdowns) * 100
    
    # Trade statistics
    num_trades = len(trade_logs)
    total_fees_paid = sum(log[4] for log in trade_logs)
    
    print("\n" + "="*80)
    print("PORTFOLIO ENGINE AUDIT & BACKTEST PERFORMANCE METRICS")
    print("="*80)
    print(f"  - Initial Capital:        ${initial_capital:,.2f} USD")
    print(f"  - Final Capital:          ${equity_curve[-1]:,.2f} USD")
    print(f"  - Triton Fuser Return:    {total_returns:+.4f}%")
    print(f"  - Benchmark Return:       {benchmark_returns:+.4f}%")
    print(f"  - Total Trades Executed:  {num_trades} trades")
    print(f"  - Total Exchange Fees:    ${total_fees_paid:,.2f} USD")
    print(f"  - Annualized Sharpe:      {sharpe:.4f}")
    print(f"  - Maximum Drawdown:       {max_dd:.4f}%")
    
    if total_returns > benchmark_returns:
        print(f"  - ALPHA STATUS:           SUCCESS (Outperformed Benchmark by {total_returns - benchmark_returns:.4f}%)")
    else:
        print("  - ALPHA STATUS:           UNDERPERFORMED")
    print("="*80)

    # 5. Generate and Save Performance Visualizations
    print("[Backtest Graphics] Generating portfolio equity performance plots...")
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve / 1e6, label='Triton Fuser (Hydrodynamic Alpha)', color='#10b981', linewidth=2.5)
    plt.plot(benchmark_curve / 1e6, label='Benchmark (Buy & Hold)', color='#6b7280', linestyle='--', linewidth=1.5)
    plt.title('Triton Fuser: Microsecond-Latency Equity Backtest Audit', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('HFT Snapshot Steps', fontsize=12)
    plt.ylabel('Portfolio Equity (Millions USD)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')
    
    # Save diagram
    save_path = "backtest_performance.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Backtest Graphics] Beautiful performance curve saved successfully to '{save_path}'!")

if __name__ == "__main__":
    run_hft_backtester()
