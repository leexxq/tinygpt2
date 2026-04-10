
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn 
from torch.nn import functional as F

import math

import tiktoken

class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj =  nn.Linear(config.n_embd,config.n_embd)

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size))\
                             .view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.size()
        qkv : torch.Tensor = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim = 2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
        q = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
        v = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) #(B,nh,T,T)

        att = att.masked_fill(self.get_buffer('bias')[:,:,:T,:T] == 0,float('-inf'))

        
        att = F.softmax(att,dim=-1)

        y = att @ v #(B,nh,T,T) @ (B,nh,T,ns) -> (B,nh,T,ns)
        y = y.transpose(1,2).contiguous().view(B,T,C) # (B,nh,T,ns) -> (B,T,nh,ns) -> (B,T,C)
        y = self.c_proj(y)
        
        return y






class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.c_fc =  nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self,x):
        self.c_fc(x)
        self.gelu(x)
        self.c_proj(x)

class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int =6
    n_head: int = 6
    n_embd: int = 354

class GPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size,config.n_embd),
                wpe = nn.Embedding(config.block_size,config.n_embd),
                h = nn.ModuleList(Block(config) for _ in range(config.n_layer) ),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size ,bias=True)

    def forward(self,idx:torch.Tensor):
        B,T = idx.size()
        assert T <= self.config.block_size , f"connot forward sequence of lenth{T},block size is {self.config.block_size}"

        pos = torch.range(0,T,dtype= torch.long,device=idx.device)

        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        # if 'dropout' in override_args:
        #     print(f"overriding dropout rate to {override_args['dropout']}")
        #     config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        #print(sd_keys)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        #print(sd_keys_hf)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 




num_return_sequences = 5
max_length = 30

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"

print("using device:",device)


with open('data/shakespeare/input.txt','r') as f:
    text = f.read()

data = text[:1000]

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
B,T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B,T)
y = buf[1:].view(B,T)


model = GPT(GPTConfig())
model.to(device=device)
logits:torch.Tensor = model(x)

print(logits.shape)





# while x.size(1) < max_length:
#     pass


