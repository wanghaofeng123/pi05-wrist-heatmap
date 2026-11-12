"""Training-only wrist heatmap supervision. No robot or inference dependencies."""
import torch
from torch import nn
import torch.nn.functional as F

class SpatialHead(nn.Module):
    def __init__(self, width=2048):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.GELU(),nn.Linear(128,1))
    def forward(self, prefix):
        # Prefix image order: base, left wrist, masked right wrist, then language.
        assert prefix.shape[1]>=768
        x=prefix[:,256:512].to(next(self.parameters()).dtype)
        return self.net(x).squeeze(-1).float()

def spatial_loss(logits, uv, valid):
    uv=torch.as_tensor(uv,device=logits.device,dtype=torch.float32)
    valid=torch.as_tensor(valid,device=logits.device,dtype=torch.bool)
    y,x=torch.meshgrid(torch.arange(16,device=logits.device),torch.arange(16,device=logits.device),indexing='ij')
    # uv already refers to 224x224 padded input, pixel-center convention.
    q=(uv+.5)/14-.5
    target=torch.exp(-((x[None]-q[:,0,None,None])**2+(y[None]-q[:,1,None,None])**2)/(2*.75**2)).flatten(1)
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
    per=-(target*F.log_softmax(logits,dim=-1)).sum(-1)
    count=valid.sum().float();world=1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world=torch.distributed.get_world_size();torch.distributed.all_reduce(count)
    return (per*valid).sum()*world/count.clamp_min(1)
