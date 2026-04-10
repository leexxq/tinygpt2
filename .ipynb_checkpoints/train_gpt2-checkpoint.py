import torch
import torch.nn as nn 
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int =6
    n_head: int = 6
    n_embd: int = 354

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transfomer = nn.MouduleDict()