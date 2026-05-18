import struct
import time

def benchmark_python(filename):
    print("Starting Python benchmark...")
    
    # Initialize mock order book
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    volumes = [0] * 8
    
    start_time = time.time()
    
    with open(filename, 'rb') as f:
        data = f.read()
        
    num_messages = len(data) // 8
    
    # Python parsing loop
    for i in range(num_messages):
        offset = i * 8
        target_p, delta_v = struct.unpack_from('ii', data, offset)
        
        # update book
        for j in range(8):
            if prices[j] == target_p:
                volumes[j] += delta_v
                
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Python took {duration:.4f} seconds for {num_messages} messages.")
    print(f"Time per message: {(duration / num_messages) * 1e6:.4f} microseconds.")

if __name__ == '__main__':
    benchmark_python('historical_itch_feed.bin')
