# scripts/generate_dummy_data.py
import struct
import numpy as np

NUM_MESSAGES = 100000

def generate_dummy_data(filename='historical_itch_feed.bin'):
    print(f"[Simulator] Generating {NUM_MESSAGES} dynamic, high-frequency L2 ticks...")
    
    # We will simulate a mean-reverting price process with trending regimes
    prices = []
    deltas = []
    
    current_mid = 104.0
    drift = 0.0001
    
    for i in range(NUM_MESSAGES):
        # Apply structured trending waves
        if (i // 10000) % 2 == 0:
            current_mid += np.random.normal(0.005, 0.08)  # Buying wave
        else:
            current_mid += np.random.normal(-0.005, 0.08) # Selling wave
            
        # Target price bounded to [100, 108]
        current_mid = max(100, min(108, current_mid))
        
        # Populate limit order bids and asks around the moving mid price
        target_p = int(np.round(current_mid + np.random.choice([-1.5, -0.5, 0.5, 1.5])))
        
        # Volumetric imbalance shift
        if (i // 10000) % 2 == 0:
            delta_v = np.random.randint(-2, 15)  # Volumetric accumulation (bids increasing)
        else:
            delta_v = np.random.randint(-15, 2)  # Volumetric liquidation (asks piling)
            
        prices.append(target_p)
        deltas.append(delta_v)
        
    prices = np.array(prices, dtype=np.int32)
    deltas = np.array(deltas, dtype=np.int32)
    
    with open(filename, 'wb') as f:
        for p, d in zip(prices, deltas):
            f.write(struct.pack('ii', p, d))
    print(f"[Simulator] Simulation successful. Saved dynamic feed to '{filename}'.")

if __name__ == '__main__':
    generate_dummy_data()
