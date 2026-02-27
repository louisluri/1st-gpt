import torch
import torch.nn as nn
from torch import arange
from torch.nn import functional as F
import gc

def sample_logits(logits, temp=1.0, top_k=0, top_p=0.0):
    logits /= temp

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of attention in paralel"""
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout_p = dropout
        self.head_size = n_embed // n_head
        self.qkv_projection = nn.Linear(n_embed, n_embed*3, bias=False)
        self.proj = nn.Linear(self.head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

        # torch.arange(0, D, 2) gets [0, 2, 4, ..., (D - 2)]
        inv_freq = 1.0 / (float(10000.0) ** (arange(0, self.head_size, 2).float() / self.head_size))  # 1 / (base^(2i/D))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(self.n_embed, dim=2)  # (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        q = q.to(torch.float32)
        k = k.to(torch.float32)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            inv_freq = self.inv_freq.to(torch.float32)
            t = torch.arange(T, device=x.device, dtype=torch.float32)
            
            freqs = torch.einsum("i,j->ij", t, inv_freq) # [T, head_size/2]
            emb = torch.cat((freqs, freqs), dim=-1) # (T, head_size)
            
            cos = emb.cos()[None, None, :, :].to(torch.float32) # [B, nh, T, hs]
            sin = emb.sin()[None, None, :, :].to(torch.float32)
            #print(f"({k[13][12][100][4]}, {k[13][12][100][(head_size//2)+4]})")
            #print(inv_freq[4])
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            #print(f"({k[13][12][100][4]}, {k[13][12][100][(head_size//2)+4]})")

        q = q.to(v.dtype) 
        k = k.to(v.dtype)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embed)
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(nn.Module):
    """Feed Forward Block"""
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), 
            nn.GELU(), 
            nn.Linear(4 * n_embed, n_embed), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, n_head, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.RMSNorm(n_embed)
        self.ln2 = nn.RMSNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, index, targets=None):
        B, T = index.shape
        
        x = self.token_embedding_table(index) # (B,T,C)
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, index, max_new_tokens, temp=1.0, top_k=0, top_p=0.0, eot_id=None):
        with torch.no_grad():
            self.eval()
            for _ in range(max_new_tokens):
                # get the logits for each char in the index
                logits, loss = self(index)
                # only look at the logits for each last letter
                logits = logits[:, -1, :]
                # choose one sample from our logits
                index_next = sample_logits(logits, temp, top_k, top_p)
                #concatenate the letter choice for each batch onto the end of the existing char list
                index = torch.cat((index, index_next), dim=1) #(B, T+1)
                # stop generating if end of text token is generated
                if eot_id is not None and index_next.item() == eot_id:
                    break
            self.train()
        return index


# SAVE/LOAD CHECKPOINT FUNCTIONS
def load_checkpoint(model, optimizer, scheduler, path):
    """loads a previous checkpoint from the checkpoint path specified above, 
    returns the most recent optimizer step that model and optimizer were saved on"""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model']) 
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step, epoch, block = checkpoint['meta']['step'], checkpoint['meta']['epoch'], checkpoint['meta']['block']
    del checkpoint
    gc.collect() 
    torch.cuda.empty_cache()
    return step, epoch, block

def save_checkpoint(step, epoch, block, model, optimizer, scheduler, path):
    """saves the current model and optimizer state to the checkpoint path specified at the top
    Prints to confirm completion"""
    raw_state = model.state_dict()
    clean_state = {}

    for k, v in raw_state.items():
        if k.startswith("_orig_mod."):
            clean_k = k.replace("_orig_mod.", "")
        else:
            clean_k = k
        clean_state[clean_k] = v

    optim_state = optimizer.state_dict()
        
    checkpoint = {
            'model': clean_state,
            'optimizer': optim_state,
            'scheduler_state_dict': scheduler.state_dict(),
            'meta': {'step': step, 'epoch': epoch, 'block': block}
            }
    # Save the dictionary to a file. overwrites old checkpoint
    torch.save(checkpoint, path)
    del checkpoint, clean_state, optim_state
    gc.collect() 
    torch.cuda.empty_cache()
    print(f"Checkpoint (step: {step}, epoch: {epoch}, block: {block}) Saved")