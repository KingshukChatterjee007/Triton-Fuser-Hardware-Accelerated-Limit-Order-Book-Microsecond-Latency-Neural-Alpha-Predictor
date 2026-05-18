# Triton Fuser: Continuous Order Book Density Modeling via PINNs

**Triton Fuser** is a systems research prototype exploring the representation of Limit Order Book (LOB) depth as a continuous fluid density field $\rho(x, t)$ governed by a 1D Advection-Diffusion PDE:

$$\frac{\partial \rho}{\partial t} + u \frac{\partial \rho}{\partial x} = D \frac{\partial^2 \rho}{\partial x^2}$$

This repository is designed as a **high-throughput historical research simulator and mathematical explorer** rather than live, production-grade HFT infrastructure. It serves as an experimental bridge between vectorized C++ log-parsing, PyTorch-based Physics-Informed Neural Network (PINN) training, and autotuned OpenAI Triton GPU kernels.

> [!NOTE]
> **Repository Name Acknowledgment**: The local directory name (`Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor`) is a descriptive experimental path. If hosting this repository on GitHub, we highly recommend renaming the remote repository to a clean, standard identifier such as `triton-fuser` or `fluid-lob`.

---

## 1. Project Framing & Realistic Scope

In systems engineering, claiming "microsecond end-to-end latency" for deep-learning-driven trading setups requires extreme caveats. Triton Fuser is presented honestly:

1. **Not a Live Trading System**: The codebase does not include kernel bypass network stacks (e.g., Solarflare EF_VI or OpenOnload), DPDK packet processing, hardware feed-handlers (FPGA), or strict NUMA-node socket mapping. The live socket listener (`run_live_mode` inside [ingestion_engine.cpp](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/ingestion_engine.cpp)) is a basic Windows Winsock/BSD UDP prototype and should not be used in live production environments without complete systems rewrites.
2. **Research & Backtesting Simulator**: The C++ and Python pipeline is designed for **high-throughput historical tick processing, backtesting simulation, and neural model training**. It is optimized to stream through gigabytes of raw Level 2 binary tick logs and fit numerical fields efficiently.

---

## 2. Technical Analysis: The GPU/Triton Inference Justification

A common point of skepticism in low-latency systems is the integration of GPU inference (via Triton or standard PyTorch CUDA) on individual order book tick updates. In production HFT environments, **GPU inference is generally avoided on the hot-path** for several reasons:

### Latency vs. Throughput Trade-Offs

```
                       LATENCY REGIME                               THROUGHPUT REGIME
             (Single-Tick Live Execution)                      (Historical Simulation / Research)

   [Tick Arrival]                                       [Batch of 4,096+ Snaps]
         │                                                        │
   ┌─────┴─────────────────────────────┐                    ┌─────┴─────────────────────────────┐
   │ Host CPU                          │                    │ Host CPU                          │
   │ (Thread-pinned, NUMA local)       │                    │ (Batched marshalling via Ctypes)   │
   └─────┬─────────────────────────────┘                    └─────┬─────────────────────────────┘
         │ (Sub-microsecond direct execution)                     │ (High-bandwidth PCIe DMA Transfer)
         ▼                                                        ▼
   [Predictive Output] (Sub-10µs)                       ┌───────────────────────────────────────┐
                                                        │ GPU Device (SRAM tiled execution)     │
                                                        │ (Triton fused matmul + Softplus)      │
                                                        └─────────────────┬─────────────────────┘
                                                                          ▼
                                                                  [Batched Output] (Sub-millisecond)
```

* **PCIe Bus Latency**: Transferring a small feature tensor (e.g., $1 \times 100$ bins) from Host memory to Device memory via PCIe, launching the CUDA kernel, and copying the result back introduces a minimum roundtrip latency of **5 to 15 microseconds**. This overhead is an order of magnitude slower than a simple C++ linear regression or decision tree evaluated locally on a single CPU core.
* **Batching Bottleneck**: Deep learning models require batching to saturate GPU tensor cores. Waiting to accumulate a batch of ticks in a live trading environment introduces queueing jitter, which is unacceptable in latency-sensitive market-making.

### Where the Triton GPU Kernel is Justified
The custom OpenAI Triton kernel ([fused_kernel.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/fused_kernel.py)) is integrated for the **high-throughput research and simulation regime**:
* **Off-line Simulation**: When backtesting over months of historical L2 tick feeds or evaluating cross-sectional signals across multiple tickers, the batch size is extremely large ($B \ge 4096$).
* **Arithmetic Intensity**: In this regime, the parallel capability of the GPU dominates. The autotuned Triton kernel merges matrix multiplication, bias addition, and a numerically stable Softplus activation (`tl.maximum(0.0, x) + tl.log(1.0 + tl.exp(-tl.abs(x)))`) directly into the SRAM register-tiling stage. This avoids high-bandwidth memory (HBM) write-backs between mathematical operations, reducing execution latency for large batch sizes.

---

## 3. System Architecture

The pipeline processes historical tick files via the following stages:

```
  [Binary Tick Log (.bin)] ──► [Memory Mapping (mmap/MapViewOfFile)]
                                            │
                                            ▼
                                [C++ Ingestion Engine] (AVX2 SIMD Gather Binning)
                                            │
                                            ▼ (ctypes / zero-copy buffers)
                                [Python Data Bridge] (Adaptive width, sparsity gates)
                                            │
                                            ▼
                               [PyTorch Neural PDE PINN] (Coordinates [x, t] -> Rho)
                                            │
                                            ▼ (Large batches only)
                              [OpenAI Triton Fused Kernel] (SRAM Tiled MatMul + Activation)
```

---

## 4. Component Breakdown & Implementation

### 1. Ingestion Engine
**Source Code:** [ingestion_engine.cpp](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/ingestion_engine.cpp)

A C++ utility designed to parse binary L2 tick logs with high throughput.
* **Memory Mapping**: Uses `MapViewOfFile` (Windows) or `mmap` (Linux) to map tick files directly into the virtual address space. This avoids user-space copy buffers and allows the OS page cache to manage memory staging.
* **AVX2 SIMD Vectorization**: The binner (`process_empirical_data`) loads 8 tick price/volume updates simultaneously. It uses SIMD register gathers (`_mm256_i32gather_epi32`) and vector float divisions (`_mm256_div_ps`) to bin ticks relative to the mid-price in parallel.
* **Vectorized Book Updates**: Provides an auxiliary `update_book_simd` function demonstrating how 8 price levels can be loaded, compared to a target price using `_mm256_cmpeq_epi32` registers, and updated without hot-path memory allocations.

### 2. Python-Ctypes Bridge
**Source Code:** [python_bridge.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/python_bridge.py)

Loads the compiled ingestion library using Python's standard `ctypes` wrapper.
* **Buffer Sharing**: Allocates raw flat arrays in Python and passes their pointers to the C++ parser. The returned data is wrapped directly as NumPy arrays and PyTorch tensors without intermediate memory copying.
* **Adaptive Bin Width**: Gauges rolling bid/ask spreads to dynamically scale bin sizes, preventing spatial discretization errors across changing volatility regimes.
* **Sparsity Guard**: Checks the ratio of empty spatial bins in the LOB. If the order book is too sparse, it disarms processing to prevent noisy derivatives.

### 3. Physics-Informed Neural Network (PINN)
**Source Code:** [physics_engine.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/physics_engine.py)

Models LOB volume depth as a continuous spatial-temporal field, regularized by fluid dynamics.
* **Continuum Modeling**: Evaluates coordinate vectors $[x, t]$ (price offset and time) to predict the liquid density $\rho(x, t)$.
* **PDE Residual Formulation**: Uses PyTorch's autograd engine (`torch.autograd.grad`) to calculate continuous derivatives:
  $$\text{Residual} = \frac{\partial \rho}{\partial t} + u \frac{\partial \rho}{\partial x} - D \frac{\partial^2 \rho}{\partial x^2}$$
* **Joint Loss**: Optimizes network weights against empirical data MSE while minimizing the continuous PDE residual on random collocation points in the domain:
  $$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{PDE}}$$
  The advection drift $u$ and diffusion viscosity $D$ are learned dynamically as model parameters.

### 4. JIT-Compiled Triton GPU Kernel
**Source Code:** [fused_kernel.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/fused_kernel.py)

An OpenAI Triton Python implementation compiling a custom CUDA kernel at runtime.
* **Operator Fusion**: Fuses the final fully-connected layer matrix projection, bias addition, and a stable Softplus activation (`tl.maximum(0.0, x) + tl.log(1.0 + tl.exp(-tl.abs(x)))`) into a single kernel.
* **Register Tiling**: Performs block accumulation in SRAM registers to bypass global HBM read/write bottlenecks.
* **Autotuning Configs**: Employs `@triton.autotune` configurations to find optimal block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), stage pipeline depths, and warps for the current GPU accelerator.

---

## 5. Performance Audits & Benchmarking

The repository contains several scripts to evaluate different aspects of the pipeline:

1. **Binance Live Stream Scraper ([fetch_binance_l2.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/scripts/fetch_binance_l2.py))**
   - Connects to the Binance L2 WebSocket diff-depth API, fetches initial REST L2 state snapshots, updates rolling L2 bids/asks, and packages the feed into binary files using C-compatible integer layouts (`struct.pack('ii')`).
2. **End-to-End Pipeline Audit ([end_to_end_pipeline.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/scripts/end_to_end_pipeline.py))**
   - Integrates the entire sequence (C++ tick parsing -> PyTorch tensor loading -> Train/OOS split -> joint PINN fit -> OOS evaluation -> rolling Alpha indicator correlation -> performance profile).
   - Computes Alpha signal metrics: correlates the learned advection velocity $u$ (fluid center-of-mass drift rate) against future price returns to calculate an Information Coefficient (IC) and directional accuracy.
   - Measures end-to-end hot path execution latency over 50 iterations.
3. **Stochastic Bootstrap Ablation Study ([academic_ablation_study.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/scripts/academic_ablation_study.py))**
   - Compares the Physics-PINN regularized model against a baseline coordinate MLP optimized purely for spatial data MSE.
   - Implements a **PDE Residual Noise Filter**: uses the local physical residual $|R(x, t)|$ as an indicator check. If the residual exceeds a threshold, the model flags the state as high-entropy non-equilibrium and skips prediction.
   - Computes directional hit rates and prediction coverage across multiple seeds using a robust 85% statistical bootstrap resampling with replacement to evaluate variance.
4. **Regime Portfolio Stress Backtester ([multi_stock_backtest.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/scripts/multi_stock_backtest.py))**
   - Stress-tests trading decisions using the rolling $u$-momentum signal across 5 simulated stock feeds experiencing distinct volatility/drift regimes (NVDA, TSLA, AAPL, MSFT, AMZN).
   - Incorporates realistic market friction constants: transaction fees (0.5 bps maker/taker), execution delays (simulating latency lags), and dynamic volatility-scaled position sizing.

---

## 6. Honest Technical Limitations & Production Bottlenecks

In the spirit of honest systems engineering, several real-world bottlenecks are present:

1. **Host-to-Device Memory Transfer (PCIe)**
   - Copying tensors from CPU host memory to GPU device memory is slow for small batch sizes. For individual tick inference, PCIe latency completely negates any speedups from the Triton kernel. The GPU pipeline only becomes efficient for batch sizes larger than $512$.
2. **Python ctypes Overhead**
   - While the compiled C++ AVX2 binner runs in nanoseconds, Python's runtime, `ctypes` marshalling, and PyTorch tensor instantiation introduce **~2.5 to 3 milliseconds** of latency per loop. For actual microsecond execution, the entire inference engine must be re-implemented in C++ (e.g., using LibTorch to run models directly in the feed handler thread).
3. **No Kernel Bypass or Hardware Integration**
   - This prototype does not implement kernel-bypass network stacks (such as Solarflare EF_VI or OpenOnload), DPDK socket mapping, or hardware-based feed parsing (FPGA). It runs over standard OS network sockets and filesystem drivers.

---

## 7. Environment Setup & Build Guide

### Prerequisites
- **Compiler**: MSVC on Windows (via Developer Command Prompt) or GCC/G++ on Linux.
- **Python**: Version 3.10+ with PyTorch installed.
- **CUDA Toolkit**: Version 11.8+ (strictly required if compiling Triton JIT kernels on Linux).

### 1. Compile the Ingestion Engine
- **On Windows (MSVC Developer Command Prompt)**:
  ```bash
  cl.exe /O2 /arch:AVX2 /LD src/ingestion_engine.cpp /Fe:src/ingestion_engine.dll
  ```
- **On Linux (GCC)**:
  ```bash
  g++ -O3 -mavx2 -shared -fPIC src/ingestion_engine.cpp -o src/ingestion_engine.dll
  ```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Validation Pipeline
To run the full C++ ingestion, tensor bridge, and PINN fitting loop:
- **On Windows PowerShell**:
  ```powershell
  $env:PYTHONPATH="src;."
  python -m scripts.end_to_end_pipeline
  ```
- **On Linux / WSL Bash**:
  ```bash
  PYTHONPATH=src:. python scripts/end_to_end_pipeline.py
  ```

### 4. Run Ablation and Backtest Audits
To execute the peer-review bootstrap ablation study and the multi-regime backtester:
- **On Windows PowerShell**:
  ```powershell
  python -m scripts.academic_ablation_study
  python -m scripts.multi_stock_backtest
  ```
- **On Linux / WSL Bash**:
  ```bash
  PYTHONPATH=src:. python scripts/academic_ablation_study.py
  PYTHONPATH=src:. python scripts/multi_stock_backtest.py
  ```
