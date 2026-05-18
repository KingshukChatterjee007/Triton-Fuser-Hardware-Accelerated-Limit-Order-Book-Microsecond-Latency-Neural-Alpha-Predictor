import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_physics_alpha_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Determine execution coordinates for the current block
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute memory offsets for SRAM tiling
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Vectorized pointer tracking initialization
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Initialize accumulation register in local SRAM cache
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate across matrix inner dimension blocks
    for k in range(0, K, BLOCK_K):
        x_block = tl.load(x_ptrs)
        w_block = tl.load(w_ptrs)
        
        # Core Matrix Multiplication step
        accumulator += tl.dot(x_block, w_block)
        
        # Advance pointers along inner dimension dimension
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Load bias vectors directly into local execution scope
    b_ptrs = B_ptr + offs_n[None, :]
    bias = tl.load(b_ptrs)
    accumulator = accumulator + bias

    # ── FUSED ACTIVATION: Continuous Softplus Physics Anchor ──
    # Math: stable ln(1 + exp(x)) = max(0, x) + ln(1 + exp(-|x|)) to prevent overflow to infinity
    accumulator = tl.maximum(0.0, accumulator) + tl.log(1.0 + tl.exp(-tl.abs(accumulator)))

    # Determine outbound destination pointers
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    
    # Store with masks for out of bounds checking if M/N aren't perfect multiples of blocks
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=mask)

def run_fused_alpha_layer(x, w, b):
    M, K = x.shape
    K, N = w.shape
    
    # Asserting types and contiguousness
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert b.is_contiguous()
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    fused_physics_alpha_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        out.stride(0), out.stride(1)
    )
    return out

if __name__ == "__main__":
    print("Initializing Fused Triton Kernel Benchmark...")
    
    if torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        device = 'cuda' if torch.cuda.is_available() else 'xpu'
        print(f"Using device: {device}")
        
        M, N, K = 1024, 1024, 1024
        x = torch.randn((M, K), device=device, dtype=torch.float32)
        w = torch.randn((K, N), device=device, dtype=torch.float32)
        b = torch.randn((N,), device=device, dtype=torch.float32)
        
        # Test correctness vs PyTorch
        out_triton = run_fused_alpha_layer(x, w, b)
        out_torch = torch.nn.functional.softplus(torch.matmul(x, w) + b)
        
        max_diff = torch.max(torch.abs(out_triton - out_torch))
        print(f"Max difference between PyTorch and Triton: {max_diff.item()}")
        
        # Profiling
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['M', 'N', 'K'],
                x_vals=[128, 256, 512, 1024],
                line_arg='provider',
                line_vals=['triton', 'torch'],
                line_names=['Triton', 'Torch'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='GB/s',
                plot_name='matmul-performance',
                args={}
            )
        )
        def benchmark(M, N, K, provider):
            x = torch.randn((M, K), device=device, dtype=torch.float32)
            w = torch.randn((K, N), device=device, dtype=torch.float32)
            b = torch.randn((N,), device=device, dtype=torch.float32)
            
            quantiles = [0.5, 0.2, 0.8]
            
            if provider == 'torch':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.softplus(torch.matmul(x, w) + b), quantiles=quantiles)
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_fused_alpha_layer(x, w, b), quantiles=quantiles)
            
            # Simple GB/s calculation
            gbps = lambda ms: 2 * M * N * K * 1e-9 / (ms * 1e-3)
            return gbps(ms), gbps(max_ms), gbps(min_ms)
        
        benchmark.run(print_data=True, show_plots=False)
        
    else:
        print("Triton requires an accelerator (CUDA or XPU). Found none, so skipping benchmark execution.")
