from dataclasses import dataclass

import torch
import torch.nn as nn 
        
from torch.nn import functional as F

import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # dropout: float = 0.0
    # bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    

class DataLoaderLite:
    def __init__(self,B,T) :
        self.B = B
        self.T = T
        with open('data/shakespeare/input.txt','r') as f:
            text = f.read()
        try:
            import tiktoken
            enc = tiktoken.get_encoding('gpt2')
        except :
            import tokenizer.encoder
            enc = tokenizer.encoder.get_encoder('tokenizer')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        self.current_position = 0

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T

        if self.current_position + (B*T + 1) > len(self.tokens): 
            self.current_position = 0

        return x,y





class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.register_buffer('scale_init',None,persistent=False)

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size))\
                             .view(1,1,config.block_size,config.block_size))


        if hasattr(F,'scaled_dot_product_attention') :
            self._sdpa = F.scaled_dot_product_attention;
        else:
            self._sdpa = CausalSelfAttention._scaled_dot_product_attention;

    # Efficient implementation equivalent to the following:
    @staticmethod
    def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(device=query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(self,x):
        B,T,C = x.size()
        qkv : torch.Tensor = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim = 2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)

        # attention 
        # att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) #(B,nh,T,T)
        # att = att.masked_fill(self.get_buffer('bias')[:,:,:T,:T] == 0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v #(B,nh,T,T) @ (B,nh,T,ns) -> (B,nh,T,ns)
        # flash attention
        
        y = self._sdpa(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C) # (B,nh,T,ns) -> (B,T,nh,ns) -> (B,T,C)
        y = self.c_proj(y)
        
        return y


class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.c_fc =  nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.register_buffer('scale_init',None,persistent=False)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

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
        return x


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
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size ,bias=False)

        assert isinstance(self.transformer.wte ,nn.Embedding)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self.__init_weights)

    def __init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'scale_init'):
                std *= (2 * self.config.n_layer)** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)


    def configure_optimizer(self,weight_decay,learning_rate,device) -> torch.optim.AdamW:
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
        return optimizer


    def forward(self,idx:torch.Tensor,targets = None):
        B,T = idx.size()
        assert T <= self.config.block_size , f"connot forward sequence of lenth{T},block size is {self.config.block_size}"
        pos = torch.arange(0,T,dtype= torch.long,device=idx.device)

        assert isinstance(self.transformer.wpe,nn.Embedding)
        assert isinstance(self.transformer.wte,nn.Embedding)
        pos_emb = self.transformer.wpe(pos) 
        tok_emb = self.transformer.wte(idx) 

        x = pos_emb + tok_emb

        assert isinstance(self.transformer.h,nn.ModuleList)
        for block in self.transformer.h:
            x = block(x)

        assert isinstance(self.transformer.ln_f, nn.LayerNorm)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) #(B,T,vocab_size)
        #logits.view(-1,logits.size(-1)) : (B*T,vocab_size)
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
            pass

        return logits, loss

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


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_step = 50

max_batch_size = 16 * 256 
B = 8
T = 256
grad_accum_steps= max_batch_size//(B*T)

num_return_sequences = 5
max_length = 30

def get_lr(iter):
    if iter < warmup_steps: 
        return max_lr * (iter + 1) /warmup_steps

    if iter > max_step:
        return max_lr
    
    decay_ratio = (iter - warmup_steps) /(max_step - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi*decay_ratio))
    return min_lr + coeff*(max_lr - min_lr)



torch.manual_seed(1337)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(1337)
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"
    torch.mps.manual_seed(1337)

print("using device:",device)

train_loader = DataLoaderLite(B=B,T=T)

# torch.set_float32_matmul_precision('medium')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device=device)
# model = torch.compile(model)

# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1,learning_rate=6e-4,device=device)

import time
for step in range(max_step):
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device=device),y.to(device=device)
        # with torch.autocast(device_type=device,dtype=torch.bfloat16):
        #     logits ,loss = model(x,y)
        logits ,loss = model(x,y)
        assert isinstance(loss,torch.Tensor)
        loss = loss / grad_accum_steps #ensure accum loss equal to origin loss
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f"step{step},loss:{loss_accum:.3f},lr:{lr:.6f},norm:{norm:.4f},dt:{dt:.2f}ms")
