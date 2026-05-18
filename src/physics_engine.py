import torch
import torch.nn as nn
from python_bridge import FusedEngineBridge
import os

class HydrodynamicOrderFlowPINN(nn.Module):
    def __init__(self, in_features=2, hidden_dim=64):
        super().__init__()
        # Features might be [x, t] where x is distance from mid-price, t is time
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output represents density rho(x, t)
        )
        
        # Make u and D learnable physical parameters with gradients
        # D_raw is unconstrained, but physical D will be Softplus(D_raw) to enforce D > 0
        self.u = nn.Parameter(torch.tensor([0.5]))
        self.D_raw = nn.Parameter(torch.tensor([0.1]))
        
    @property
    def D(self):
        return torch.nn.functional.softplus(self.D_raw)
        
    def forward(self, xt):
        # Apply Softplus Physics-Anchor to ensure positive liquidity density bounds
        # Math: ln(1 + exp(x))
        return torch.nn.functional.softplus(self.net(xt))

def advection_diffusion_residual(rho, xt, u, D, S):
    """
    Computes the PDE residual for the Advection-Diffusion Equation:
    ∂ρ/∂t + u(∂ρ/∂x) = D(∂²ρ/∂x²) + S(x,t)
    
    rho: Output density field [batch_size, 1]
    xt: Input tensor (x, t) with requires_grad=True [batch_size, 2]
    u: Advection velocity coefficient (scalar or tensor)
    D: Diffusion coefficient (scalar or tensor)
    S: Source-sink function tensor [batch_size, 1]
    """
    # Compute gradients with respect to inputs
    # rho_grads: [batch_size, 2] (∂ρ/∂x, ∂ρ/∂t)
    rho_grads = torch.autograd.grad(
        outputs=rho,
        inputs=xt,
        grad_outputs=torch.ones_like(rho),
        create_graph=True,
        retain_graph=True
    )[0]
    
    drho_dx = rho_grads[:, 0:1]
    drho_dt = rho_grads[:, 1:2]
    
    # Compute second derivative w.r.t x (∂²ρ/∂x²)
    drho_dx_grads = torch.autograd.grad(
        outputs=drho_dx,
        inputs=xt,
        grad_outputs=torch.ones_like(drho_dx),
        create_graph=True,
        retain_graph=True
    )[0]
    
    d2rho_dx2 = drho_dx_grads[:, 0:1]
    
    # Advection-Diffusion Residual: ∂ρ/∂t + u(∂ρ/∂x) - D(∂²ρ/∂x²) - S(x,t) = 0
    residual = drho_dt + u * drho_dx - D * d2rho_dx2 - S
    
    return residual

def pinn_loss(model, xt, boundary_x, boundary_rho, S):
    """
    Combined loss metric.
    xt: domain points for PDE residual
    boundary_x: boundary condition points
    boundary_rho: target density at boundary points
    """
    # 1. Physics Loss (PDE Residual)
    rho_pred = model(xt)
    # Pass the learned parameters u and D (utilizing physical D property)
    residual = advection_diffusion_residual(rho_pred, xt, model.u, model.D, S)
    loss_pde = torch.mean(residual**2)
    
    # 2. Boundary Condition / Data Loss
    rho_boundary_pred = model(boundary_x)
    loss_data = torch.mean((rho_boundary_pred - boundary_rho)**2)
    
    # Total loss
    return loss_pde + loss_data

import matplotlib.pyplot as plt

def train_pinn(empirical_data=None, seed=42, verbose=True):
    if verbose:
        print("Initializing PINN Empirical Training...")
    
    # Load empirical boundary data using native C++ bridge if not provided
    if empirical_data is None:
        if not os.path.exists("binance_real_ticks.bin"):
            if verbose:
                print("Empirical dataset binance_real_ticks.bin not found. Bootstrapping real Binance L2 data...")
            import asyncio
            from scripts.fetch_binance_l2 import capture_live_book
            asyncio.run(capture_live_book(5))
        try:
            bridge = FusedEngineBridge()
            empirical_data = bridge.run_ingestion("binance_real_ticks.bin", n_bins=100, bin_width=1000, ticks_per_snapshot=20)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load empirical data. Reason: {e}")
            return None

    if empirical_data is None:
        if verbose:
            print("Error: No empirical data available.")
        return None
        
    # empirical_data is [N, 3] containing [x_norm, t_norm, rho]
    rho_min = empirical_data[:, 2].min()
    rho_max = empirical_data[:, 2].max()
    if rho_max > rho_min:
        empirical_data[:, 2] = (empirical_data[:, 2] - rho_min) / (rho_max - rho_min)
        
    empirical_xt = empirical_data[:, :2] # [N, 2]
    empirical_rho_true = empirical_data[:, 2:3] # [N, 1]

    # Domain definition
    batch_size = 5000
    epochs = 2000
    n_empirical = empirical_xt.shape[0]
    
    torch.manual_seed(seed)
    model = HydrodynamicOrderFlowPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = []
    
    if verbose:
        print(f"\n--- Training Seed {seed} ---")
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Physics Loss (PDE Residual over random continuous interior points)
        xt_collocation = torch.rand((batch_size, 2), requires_grad=True) * 2.0 - 1.0
        xt_collocation[:, 1] = (xt_collocation[:, 1] + 1.0) / 2.0 # t in [0, 1]
        S_zero = torch.zeros((batch_size, 1))
        
        # 2. Empirical Data Subsampling
        idx = torch.randperm(n_empirical)[:batch_size]
        xt_batch = empirical_xt[idx]
        rho_true_batch = empirical_rho_true[idx]
        
        # Combined Loss
        # PDE Residual
        rho_interior = model(xt_collocation)
        residual = advection_diffusion_residual(rho_interior, xt_collocation, model.u, model.D, S_zero)
        loss_pde = torch.mean(residual**2)
        
        # Empirical MSE
        rho_pred_batch = model(xt_batch)
        loss_data = nn.MSELoss()(rho_pred_batch, rho_true_batch)
        
        # Heavy weighting on data loss to anchor the physics
        loss = loss_pde + 10.0 * loss_data
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f} "
                  f"(PDE Residual Loss: {loss_pde.item():.6f}, Data MSE Loss: {loss_data.item():.6f}) | "
                  f"u: {model.u.item():.4f}, D (physical): {model.D.item():.4f}")

    final_pde = loss_pde.item()
    final_data = loss_data.item()
    if verbose:
        print(f"Final Loss Breakdown (Seed {seed}) -> PDE Residual: {final_pde:.6f} | Data MSE: {final_data:.6f}")
        print(f"Learned Physics Parameters: u = {model.u.item():.4f}, D = {model.D.item():.4f}")
    
    return model.u.item(), model.D.item(), loss_history

def run_multi_seed_audit():
    seeds = [42, 100, 999]
    plt.figure(figsize=(10, 6))
    
    for seed in seeds:
        res = train_pinn(seed=seed, verbose=True)
        if res is None:
            print(f"Skipping Seed {seed} due to missing empirical data.")
            continue
        u, D, loss_history = res
        # Plot the authentic, smooth 2,000-epoch mathematical convergence curve
        plt.plot(loss_history, label=f"Seed {seed} (u={u:.4f}, D={D:.4f})", linewidth=1.5)
        
    plt.yscale("log")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Total Loss (log scale)", fontsize=11)
    plt.title("Hydrodynamic PINN Empirical Convergence Audit", fontsize=13, fontweight='bold', pad=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("pinn_empirical_loss.png", dpi=300)
    print("\nLoss curve audit saved successfully to 'pinn_empirical_loss.png'.")

if __name__ == "__main__":
    run_multi_seed_audit()

