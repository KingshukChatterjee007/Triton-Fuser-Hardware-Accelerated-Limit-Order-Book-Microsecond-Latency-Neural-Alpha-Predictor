import sys
import os
import time
import asyncio
import numpy as np

# Adjust paths to import local workspace scripts and src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import capturing, parsing, and numerical engines
import scripts.fetch_binance_l2 as fetcher
from src.python_bridge import FusedEngineBridge
from src.physics_engine import train_pinn

async def main():
    print("=" * 60)
    print("STABILITY AUDIT: MULTI-WINDOW REGIME EVOLUTION")
    print("=" * 60)
    
    num_windows = 5
    window_duration = 30
    gap_duration = 30
    
    results = []
    
    for i in range(num_windows):
        print(f"\n--- WINDOW {i+1}/{num_windows} ---")
        window_filename = f"binance_window_{i+1}.bin"
        
        # Route capture output dynamically
        fetcher.OUTPUT_FILE = window_filename
        
        print(f"Capturing live Binance flow for {window_duration} seconds...")
        await fetcher.capture_live_book(window_duration)
        
        # Load snapshot data via native C++ DLL
        print("Parsing window via AVX2 C++ Engine...")
        bridge = FusedEngineBridge("src/ingestion_engine.dll")
        empirical_data = bridge.run_ingestion(window_filename, n_bins=100, bin_width=1000, ticks_per_snapshot=20)
        
        if empirical_data is None:
            print("Failed to parse data for this window.")
            continue
            
        print("Training PINN on spatial-temporal grid...")
        u, D, pde_loss, data_loss = train_pinn(empirical_data=empirical_data, seed=42, verbose=False)
        
        print(f"Learned parameters for Window {i+1}:")
        print(f"  u (Advection) = {u:.6f}")
        print(f"  D (Diffusion) = {D:.6f}")
        print(f"  Loss Residuals = PDE: {pde_loss:.6f}, Data: {data_loss:.6f}")
        
        results.append({
            'window': i + 1,
            'u': u,
            'D': D,
            'pde': pde_loss,
            'data': data_loss
        })
        
        # Clean up temporary binary files
        try:
            os.remove(window_filename)
        except OSError:
            pass
            
        if i < num_windows - 1:
            print(f"Sleeping for {gap_duration} seconds before next window...")
            time.sleep(gap_duration)
            
    print("\n" + "=" * 60)
    print("SUMMARY OF CONVERGENCE ACROSS VOLATILITY REGIMES")
    print("=" * 60)
    print(f"{'Window':<8}{'u (Advection)':<16}{'D (Diffusion)':<16}{'PDE Loss':<12}{'Data Loss':<12}")
    print("-" * 64)
    for r in results:
        print(f"{r['window']:<8}{r['u']:<16.6f}{r['D']:<16.6f}{r['pde']:<12.6f}{r['data']:<12.6f}")
    
    # Calculate variances to analyze stability
    u_vals = [r['u'] for r in results]
    D_vals = [r['D'] for r in results]
    print("-" * 64)
    print(f"Variance  u: {np.var(u_vals):.8f} | D: {np.var(D_vals):.8f}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
