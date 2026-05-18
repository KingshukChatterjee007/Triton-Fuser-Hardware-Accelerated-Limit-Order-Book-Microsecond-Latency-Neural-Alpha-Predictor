import json
import struct
import time
import urllib.request
import asyncio
import sys
import subprocess
from collections import defaultdict

try:
    import websockets
except ImportError:
    print("Installing required 'websockets' package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

# Scaling factors to fit float values into C++ int32 bounds
# BTCUSDT price is usually around 60k-100k. Tick size is 0.01.
# Scaling by 100 allows up to $21,000,000 before int32 overflow.
PRICE_SCALE = 100 
# Quantity is in BTC. Scaling by 10,000 allows up to 210,000 BTC.
QTY_SCALE = 10000 

async def capture_live_book(duration_sec=60, symbol="btcusdt", output_file="binance_real_ticks.bin"):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@depth@100ms"
    rest_url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit=1000"
    
    book_state = defaultdict(int)
    
    print(f"Fetching initial snapshot for {symbol.upper()}...")
    req = urllib.request.Request(rest_url)
    with urllib.request.urlopen(req) as response:
        snapshot = json.loads(response.read())

    print(f"Opening binary file {output_file} for writing...")
    total_ticks = 0
    with open(output_file, 'wb') as f:
        # Process snapshot and write initial liquidity depth
        def process_updates(updates):
            nonlocal total_ticks
            for price_str, qty_str in updates:
                p_int = int(float(price_str) * PRICE_SCALE)
                q_int = int(float(qty_str) * QTY_SCALE)
                
                delta_q = q_int - book_state[p_int]
                if delta_q != 0:
                    # Native C++ parsing struct mapping (target_p, delta_v)
                    f.write(struct.pack('ii', p_int, delta_q))
                    book_state[p_int] = q_int
                    total_ticks += 1

        process_updates(snapshot['bids'])
        process_updates(snapshot['asks'])
        print(f"Snapshot anchored. Tracked price levels: {len(book_state)}")

        # Connect to live WebSocket diff stream
        print(f"Connecting to Binance WebSocket stream for {duration_sec} seconds of live L2 flow...")
        start_time = time.time()
        
        async with websockets.connect(ws_url) as ws:
            messages_captured = 0
            while time.time() - start_time < duration_sec:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    
                    # 'b' for bids, 'a' for asks
                    process_updates(data.get('b', []))
                    process_updates(data.get('a', []))
                    messages_captured += 1
                    
                    if messages_captured % 50 == 0:
                        print(f"Captured {messages_captured} update events... ({(time.time() - start_time):.1f}s)")
                        
                except asyncio.TimeoutError:
                    continue
                
    print(f"Finished capturing. Gathered {total_ticks} structural liquidity updates.")
    print(f"Saved binary stream to {output_file}.")

if __name__ == "__main__":
    print("Initializing Binance L2 Capture Engine...")
    asyncio.run(capture_live_book(60))
