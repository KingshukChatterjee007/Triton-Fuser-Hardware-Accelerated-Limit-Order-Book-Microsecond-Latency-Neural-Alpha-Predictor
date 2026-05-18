import ctypes
import os
import torch
import numpy as np
import struct

class FusedEngineBridge:
    def __init__(self, dll_path=None):
        if dll_path is None:
            import sys
            ext = ".dll" if sys.platform == "win32" else ".so"
            dll_path = f"src/ingestion_engine{ext}"
        self.dll_path = os.path.abspath(dll_path)
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"Engine library not found at {self.dll_path}. Make sure to compile it for your OS first.")
        
        self.lib = ctypes.CDLL(self.dll_path)
        
        # extern "C" int process_empirical_data(const char*, float*, int, int, int, int32_t, int)
        self.lib.process_empirical_data.argtypes = [
            ctypes.c_char_p, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_int, 
            ctypes.c_int, 
            ctypes.c_int, 
            ctypes.c_int32, 
            ctypes.c_int
        ]
        self.lib.process_empirical_data.restype = ctypes.c_int

    def run_ingestion(self, tick_file, n_bins=100, bin_width=1000, max_snapshots=1000, ticks_per_snapshot=20, silent=False):
        # We need the mid_price to anchor the bins. Let's read the very first tick from the binary file.
        with open(tick_file, 'rb') as f:
            first_tick = f.read(8)
            if not first_tick:
                raise ValueError("Tick file is empty.")
            mid_price = struct.unpack('ii', first_tick)[0]
            
        b_filename = tick_file.encode('utf-8')
        
        # Allocate output buffer for density snapshots
        buffer_type = ctypes.c_float * (max_snapshots * n_bins)
        snapshot_buffer = buffer_type()
        
        if not silent:
            print(f"Streaming {tick_file} through AVX2 density engine...")
            print(f"Anchoring mid_price: {mid_price} | Bin Width: {bin_width} | Bins: {n_bins}")
        
        num_snapshots = self.lib.process_empirical_data(
            b_filename, 
            snapshot_buffer, 
            max_snapshots, 
            n_bins, 
            bin_width, 
            mid_price, 
            ticks_per_snapshot
        )
        
        if not silent:
            print(f"Extracted {num_snapshots} spatial density snapshots.")
        
        if num_snapshots == 0:
            return None, None
            
        # Convert to numpy and then to PyTorch triplets (x_normalized, t, rho)
        raw_array = np.frombuffer(snapshot_buffer, dtype=np.float32).copy()
        raw_array = raw_array.reshape((max_snapshots, n_bins))
        valid_snapshots = raw_array[:num_snapshots]
        
        empirical_data = []
        
        # Normalize x to [-1, 1], where bin = N_BINS/2 is x=0
        # Normalise t to [0, 1] across the sequence of snapshots
        for t_idx in range(num_snapshots):
            t_norm = t_idx / max(1, (num_snapshots - 1))
            for b_idx in range(n_bins):
                x_norm = (b_idx - (n_bins / 2)) / (n_bins / 2)
                rho = valid_snapshots[t_idx, b_idx]
                
                # Only keep non-zero density to keep data sparse if desired, 
                # but for PDE Data loss, we might want the zeros too. Let's keep all.
                empirical_data.append([x_norm, t_norm, rho])
                
        # Return as tensor (no autograd tracking needed on inputs)
        return torch.tensor(empirical_data, dtype=torch.float32)

if __name__ == "__main__":
    print("Testing Python-C++ Empirical Density Bridge...")
    bridge = FusedEngineBridge()
    try:
        tensor_data = bridge.run_ingestion("binance_real_ticks.bin", n_bins=100, bin_width=1000, ticks_per_snapshot=20)
        if tensor_data is not None:
            print("\nSuccessfully routed empirical density field to PyTorch tensor:")
            print(f"Shape: {tensor_data.shape} (N points x [x_norm, t_norm, rho])")
            print("First 5 rows:")
            print(tensor_data[:5])
    except FileNotFoundError as e:
        print(str(e))
