import torch
import torch.nn as nn 

DEBUG = False 

class SparseCoder(nn.Module): 
    def __init__(self, 
                 device: str, 
                 lr: float,
                 lmbd: float,
                 tol: float, 
                 dict_size: int,
                 input_dim: int, 
                 ): 
        super(SparseCoder, self).__init__()
        
        self.device = device

        self.lr = lr
        self.lmbd = lmbd
        self.tol = tol
        self.dict_size = dict_size

        self.D = nn.Parameter(torch.randn(dict_size, input_dim))
        self._init_dict()
        self._init_lr()

    def _init_dict(self): 
        with torch.no_grad(): 
            self.D.div_(torch.norm(self.D, dim=1, keepdim=True))

    def _init_lr(self): 
        with torch.no_grad(): 
            gram = torch.matmul(self.D, self.D.t())
            max_e = torch.max(torch.linalg.eigvalsh(gram)) 
            self.lr = 0.99 / max_e

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.dict_size, device=x.device)

        for i in range(1000):
            h_old = h.clone()
            
            # 1: h <= h - alpha * Dt @ (D @ h - x)
            residual = x - torch.matmul(h, self.D)
            gradient = torch.matmul(residual, self.D.t())
            h = h + self.lr * gradient
            
            # shrink(h, alpha * lambda) => shrink(a, b) = [...,sgn(a)*min(|a|-b, 0) ,...]
            h = self.shrink(h, self.lr * self.lmbd)
            
            if DEBUG: 
                if i % 100 == 0:
                    print(f"Iteration {i}:")
                    print(f"\tMax h value: {h.abs().max().item()}")
                    print(f"\tMean h value: {h.abs().mean().item()}")
                    print(f"\tResidual norm: {residual.norm().item()}")
            
            if torch.norm(h - h_old, dim=1).max() < self.tol:
                if DEBUG:
                    print(f"Converged after {i} iterations")
                break

        return h
    
    def shrink(self, a, b): 
         return torch.sign(a) * torch.clamp(torch.abs(a) - b, min=0)
    
    def reconstruct(self, h): 
        return torch.matmul(h, self.D)


if __name__ == "__main__": 
    # Simple test harness 
    input_dim = 100
    dict_size = 200
    batch_size = 32

    device = 'cpu' 
    if torch.cuda.is_available(): 
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): 
        device = 'mps'
    print(f"using device: {device}")

    sc = SparseCoder(device=device, lr=0.1, lmbd=5e-3, tol=1e-4, dict_size=dict_size, input_dim=input_dim)
    sc.to(device)

    x = torch.randn(batch_size, input_dim).to(device)

    h = sc(x)
    x_recon = sc.reconstruct(h)

    error = torch.norm(x - x_recon)
    sparsity = (h.abs() < 1e-5).float().mean()
    print(f"Reconstruction error: {error.item()}")
    print(f"Sparsity: {sparsity.item()}")