# Triton Fuser: A GPU-Accelerated Limit Order Book Density Predictor & PINN Solver

Triton Fuser is a quantitative research prototype designed to explore the continuum modeling of limit order book (LOB) liquidity depth. Instead of representing the order book as a discrete, high-dimensional list of price and volume levels, this framework models LOB volume density as a continuous spatial-temporal field, $\rho(x, t)$, governed by a 1D Advection-Diffusion partial differential equation (PDE):

$$\frac{\partial \rho}{\partial t} + u \frac{\partial \rho}{\partial x} = D \frac{\partial^2 \rho}{\partial x^2}$$

Here, $x$ represents the price offset relative to the mid-price, $t$ is time, $u$ is the learned advection drift velocity (acting as a proxy for net buy/sell order flow momentum), and $D$ is the learned diffusion coefficient (representing liquidity dispersion and spread relaxation).

The system integrates high-performance C++ ingestion, a Python-Ctypes data bridge, a Physics-Informed Neural Network (PINN) solver in PyTorch, and a custom JIT-compiled OpenAI Triton GPU kernel for hardware-fused inference.

---

## System Architecture

The following diagram illustrates the flow of order book tick updates through the system layers:

<p align="center">
  <img src="architecture_diagram.png" width="100%" alt="Triton Fuser Architecture Diagram" />
</p>

1. **Ingestion Layer (C++)**: Consumes historical binary tick files (or live UDP streams) and performs fast SIMD-accelerated binning into spatial density fields.
2. **Bridge Layer (Ctypes / Python)**: Maps the raw bin buffers into NumPy and PyTorch tensors with adaptive scaling and liquidity thresholds.
3. **Modeling Layer (PyTorch PINN)**: Solves the Advection-Diffusion PDE by training a feed-forward network to fit the observed density while regularizing predictions using automatic differentiation residuals.
4. **Hardware Fusion Layer (OpenAI Triton)**: Evaluates the neural projection layer, bias addition, and Softplus activation directly in a custom JIT-compiled GPU kernel.

---

## Component Breakdown & Implementation

### 1. Low-Latency Ingestion Engine
**Source Code:** [ingestion_engine.cpp](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/ingestion_engine.cpp)

Built in C++ to achieve high performance when processing structural order book data. The engine provides:
- **Fast Historical Parsing**: Utilizes OS-native memory-mapped files (`mmap` on POSIX systems, `MapViewOfFile` on Windows) to map raw binary tick streams directly into the virtual address space, bypassing standard user-space file buffering.
- **SIMD Binning**: Groups incoming tick updates (consisting of price and volume deltas) into discrete spatial bins relative to the rolling mid-price. Uses AVX2 register gather instructions (`_mm256_i32gather_epi32`) and vector float divisions (`_mm256_div_ps`) to bin 8 tick updates per register loop.
- **SIMD Book Updates**: Includes a vectorized function (`update_book_simd`) that loads 8 price levels and volumes into `__m256i` registers, compares them against a target price vector using register masks (`_mm256_cmpeq_epi32`), and executes a zero-allocation hot-path update.
- **Live Stream Receiver**: Embeds a basic Winsock/BSD UDP multicast socket listener (`run_live_mode`) capable of receiving L2 packet updates in real-time.

### 2. Python-Ctypes Data Bridge
**Source Code:** [python_bridge.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/python_bridge.py)

A wrapper that loads the compiled C++ shared library/DLL using Python's native `ctypes` library.
- **Zero-Copy Marshalling**: Allocates pre-aligned output float arrays in Python, passing raw C-style pointers directly into the C++ engine to receive binned snapshots. The memory is then wrapped directly as NumPy arrays and PyTorch tensors.
- **Adaptive Price Binning**: Rather than using static bin limits, it evaluates rolling price spreads in the binary feed to dynamically adjust the bin width parameter. This matches the discretization grid to the volatility regime of the asset.
- **Liquidity Sparsity Gate**: Monitors the percentage of active, non-zero price bins. If the asset density falls below a defined threshold, the engine disarms/skips processing to avoid over-trading and fee churn in illiquid conditions.

### 3. Physics-Informed Neural Network (PINN)
**Source Code:** [physics_engine.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/physics_engine.py)

Models the order book density field using a Multi-Layer Perceptron (MLP) regularized by continuous fluid equations.
- **Continuous Field Modeling**: Maps coordinate vectors $[x, t]$ (representing spatial bin distance from mid-price and time snapshot) to the predicted liquidity density $\rho(x, t)$.
- **Autograd PDE Residual**: Computes the first-order derivatives ($\frac{\partial \rho}{\partial x}$, $\frac{\partial \rho}{\partial t}$) and the second-order spatial derivative ($\frac{\partial^2 \rho}{\partial x^2}$) using PyTorch's automatic differentiation engine (`torch.autograd.grad`).
- **Learnable Physics**: The advection velocity $u$ and diffusion viscosity raw parameter $D_{raw}$ are implemented as learnable weights (`nn.Parameter`). A `Softplus` mapping enforces the physical constraint $D > 0$.
- **Joint Loss Function**: The network is trained with a combined loss:
  $$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{PDE}}$$
  Where $\mathcal{L}_{\text{data}}$ is the Mean Squared Error (MSE) against binned empirical LOB snapshots, and $\mathcal{L}_{\text{PDE}}$ is the mean squared residual of the Advection-Diffusion equation evaluated over collocation points in the coordinate space.

### 4. JIT-Compiled OpenAI Triton GPU Kernel
**Source Code:** [fused_kernel.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/src/fused_kernel.py)

To optimize model evaluation, the final forward projection layer is implemented as a custom JIT-compiled CUDA kernel using OpenAI's Triton compiler framework.
- **Operator Fusion**: Fuses matrix multiplication (representing the final weight projection), bias addition, and the structural `Softplus` activation (which enforces positive boundary bounds on predicted density $\rho$) into a single GPU hardware execution loop.
- **SRAM Tiling & Register Caching**: Tile matrices are loaded, accumulated, and modified directly within high-speed GPU SRAM registers, avoiding intermediate round-trips to the global high-bandwidth memory (HBM).
- **Numerically Stable Activation**: The fused activation implements a stable Softplus to prevent floating-point infinity overflows:
  $$\text{Softplus}(x) = \max(0, x) + \ln(1 + e^{-|x|})$$
- **Autotuning**: Uses `@triton.autotune` to evaluate different tiling sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), warp counts, and software stages to optimize execution on the target GPU architecture (e.g. NVIDIA T4).

---

## Validation & Evaluation Harnesses

The repository contains several scripts to evaluate different aspects of the pipeline:

1. **Binance Live Stream Scraper ([fetch_binance_l2.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/scripts/fetch_binance_l2.py))**
   - Connects to the Binance L2 WebSocket diff-depth API, fetches initial REST L2 state snapshots, updates rolling L2 bids/asks, and packages the feed into binary files using C-compatible integer layouts (`struct.pack('ii')`).
2. **Pipeline Integration & Profiling ([end_to_end_pipeline.py](file:///c:/Users/91704/Triton-Fuser-Hardware-Accelerated-Limit-Order-Book-Microsecond-Latency-Neural-Alpha-Predictor/scripts/end_to_end_pipeline.py))**
   - Runs a complete, integrated flow: C++ tick ingestion $\rightarrow$ Ctypes marshalling $\rightarrow$ PyTorch PINN joint training (100 epochs) $\rightarrow$ Out-of-Sample (OOS) density prediction validation.
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

## Honest Technical Limitations & Production Bottlenecks

A realistic engineering review of this research prototype reveals several bottlenecks that must be resolved for production high-frequency deployment:

1. **The Python/Ctypes Marshalling Overhead**
   - While the compiled C++ AVX2 density engine processes millions of messages at nanosecond speeds, the Python wrapper and `ctypes` serialization introduce significant overhead. Allocating standard Python floats, calling ctypes, and converting buffers to PyTorch tensors takes **~2 to 3 milliseconds** per pass.
   - **Production Resolution**: For high-frequency execution, Python must be removed from the hot-path. The inference loop must be compiled in C++ using **LibTorch** (the PyTorch C++ library). The C++ ingestion engine can then evaluate the neural network directly inside the memory-mapped threads, reducing end-to-end latency to sub-10 microseconds.
2. **WSL / OS Triton Requirements**
   - OpenAI Triton is designed primarily for Linux environments and requires a modern CUDA-compatible GPU. On Windows, Triton kernels cannot be JIT compiled unless run under WSL (Windows Subsystem for Linux) or using unofficial Windows Triton ports. `end_to_end_pipeline.py` handles this with an autoguard check that falls back to standard PyTorch matrix operators when Triton is unavailable.
3. **Sparsity Constraints in Illiquid Books**
   - Continuous PDE models assume a dense fluid medium. In illiquid stocks, L2 updates are highly sparse, resulting in many zero-filled density bins. Enforcing a differential advection-diffusion constraint on mostly empty space leads to unstable derivatives and poor convergence. The sparsity gate in the bridge layer is critical to disarm modeling under these regimes.

---

## Environment Setup & Build Guide

### Prerequisites
- **Compiler**: MSVC (via Developer Command Prompt) on Windows OR GCC/G++ on Linux.
- **Python**: Version 3.10+ with PyTorch installed.
- **CUDA Toolkit**: Version 11.8+ (strictly required if running the Triton GPU kernel on Linux).

### 1. Compile the Ingestion Engine
To build the dynamic shared library:
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

### 3. Run Pipeline Validation
To execute the complete ingestion, training, evaluation, and latency-profiling loop:
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
