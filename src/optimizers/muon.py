"""
Muon optimizer implementation.

Muon - MomentUm Orthogonalized by Newton-schulz
https://kellerjordan.github.io/posts/muon/

Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
matrix. For efficient orthogonalization we use a Newton-Schulz iteration.
"""

import torch
from torch.optim import Optimizer
from typing import Optional


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    
    Args:
        G: Matrix to orthogonalize (can be batched, ndim >= 2)
        steps: Number of Newton-Schulz iterations
        
    Returns:
        Orthogonalized matrix
    """
    assert G.ndim >= 2  # batched Muon implementation
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.bfloat16()
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, use_orthogonalization=True, use_rms=False):
    """
    Compute Muon update direction.
    
    Args:
        grad: Gradient tensor
        momentum: Momentum buffer
        beta: Momentum coefficient
        ns_steps: Number of Newton-Schulz steps
        nesterov: Whether to use Nesterov momentum
        use_orthogonalization: Whether to apply orthogonalization
        use_rms: Whether to apply RMS normalization (for ablation)
        
    Returns:
        Update direction tensor
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    
    if use_orthogonalization:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    
    # Scaling factor (original Muon behavior)
    scale = max(1, grad.size(-2) / grad.size(-1))**0.5
    
    # RMS normalization (for ablation study)
    if use_rms:
        rms = update.norm(dim=(-2, -1), keepdim=True) / (update.size(-2) * update.size(-1))**0.5
        scale = scale / (rms + 1e-7)
    
    update = update * scale
    
    return update


class Muon(Optimizer):
    """
    Muon optimizer - MomentUm Orthogonalized by Newton-schulz.
    
    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate, in units of spectral norm per update
        momentum: Momentum coefficient (default: 0.95)
        ns_depth: Number of Newton-Schulz iteration steps (default: 5)
        use_rms: Whether to use RMS normalization (for ablation study)
        use_orthogonalization: Whether to use orthogonalization (for ablation study)
        weight_decay: AdamW-style weight decay coefficient
        nesterov: Whether to use Nesterov momentum (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_depth: int = 5,
        use_rms: bool = False,
        use_orthogonalization: bool = True,
        weight_decay: float = 0.0,
        nesterov: bool = True
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_depth=ns_depth,
            use_rms=use_rms,
            use_orthogonalization=use_orthogonalization,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                # Compute Muon update
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_depth"],
                    nesterov=group["nesterov"],
                    use_orthogonalization=group["use_orthogonalization"],
                    use_rms=group["use_rms"]
                )
                
                # Apply weight decay (AdamW-style)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                
                # Apply update
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        
        return loss
    

def muon_noisy_update(param, momentum, second_momentum, beta=0.95, ns_steps=5, use_rms=False, lam=1, lr=0.02, weight_decay=0.0):
    """
    Compute Muon update direction with added Gaussian noise for exploration.
    
    Args:
        momentum: Momentum buffer
        second_momentum: Second moment buffer
        beta: Momentum coefficient
        ns_steps: Number of Newton-Schulz steps
        nesterov: Whether to use Nesterov momentum
        use_orthogonalization: Whether to apply orthogonalization
        use_rms: Whether to apply RMS normalization (for ablation)
        noise_scale: Standard deviation of Gaussian noise to add to updates
        weight_decay: Weight decay coefficient
    Returns:
        Update direction tensor
    """
    N = param.grad.numel()
    r = lam / N
    momentum.lerp_(param.grad + param * r, 1 - beta)
    momentum = param.grad.lerp_(momentum, beta)
    
    if momentum.ndim == 4:  # for the case of conv filters
        momentum = momentum.view(len(momentum), -1)
    
    momentum = zeropower_via_newtonschulz5(momentum, steps=ns_steps)

    second_momentum = beta * second_momentum + (1 - beta) * momentum * momentum

    update = momentum / (second_momentum.sqrt() + lam / N)

    scale = max(1, param.grad.size(-2) / param.grad.size(-1))**0.5

    if use_rms:
        scale = 0.2 * (update.size(-2) * update.size(-1))**0.5 / (update.norm() + 1e-7)

    param.mul_(1 - lr * weight_decay)
    param.add_(update.reshape(param.shape), alpha=-lr * scale)
    param.add_(torch.randn_like(param) / (N * second_momentum + lam).sqrt())


    
    return update

class MuonNoise(Optimizer):
    """
    MuonNoise optimizer - Muon with added Gaussian noise for exploration.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum coefficient (default: 0.95)
        ns_depth: Number of Newton-Schulz iteration steps (default: 5)
        noise_scale: Standard deviation of Gaussian noise to add to updates
        use_rms: Whether to use RMS normalization (for ablation study)
        use_orthogonalization: Whether to use orthogonalization (for ablation study)
        weight_decay: AdamW-style weight decay coefficient
        nesterov: Whether to use Nesterov momentum (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_depth: int = 5,
        use_rms: bool = False,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_depth=ns_depth,
            use_rms=use_rms,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p)
                
                # Compute Muon update
                muon_noisy_update(
                    p,
                    state["momentum_buffer"],
                    state["second_momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_depth"],
                    use_rms=group["use_rms"],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"]
                )
        
        return loss