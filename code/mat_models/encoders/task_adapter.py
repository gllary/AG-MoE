import torch.nn as nn
import torch.nn.functional as F

class TaskAdapter(nn.Module):
    """
    小型 Bottleneck Adapter，为每个任务缓冲梯度冲突
    shared_emb → Adapter(task) → head
    """

    def __init__(self, embed_dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck)
        self.nonlin = nn.SiLU()
        self.up = nn.Linear(bottleneck, embed_dim)

    def forward(self, x):
        return x + self.up(self.nonlin(self.down(x)))