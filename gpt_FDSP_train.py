import torch
import os
import time
import numpy as np
import gc
from collections import defaultdict
import bitsandbytes as bnb
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist
import functools
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import StateDictType
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing
import warnings
from gptmodel import GPTLanguageModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def master_print(*args, **kwargs):
    # LOCAL_RANK 0 is the 'leader' of the current node
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

train_dataset_path = "openwebtext1/train_data.bin"
test_dataset_path = "openwebtext1/test_data.bin"
train_map_paths = ["openwebtext1/train_shuffle_map0-128block.bin"]
val_map_path = "openwebtext1/test_shuffle_map0-128block.bin"

model_number = "04"
CHECKPOINT_PATH = f'checkpoints/model{model_number}'
LOG_FILE = f'train_data/model{model_number}_data.csv'
tokenizer_file = 'tokenizer/tokenizer-01.json'
start_epoch = 0
start_block = 0
block = start_block
max_epochs = 1
save = False   # load/save model and data?
start_optim_step = 0
optim_step = start_optim_step

# evrything here can be changed each training session to optimize learning
minibatch_size = 8   # effective batch size is minibatch_size * world_size  * accumulation_steps
accumulation_steps = 64
block_size = 128
learning_rate = 5e-3
eval_iters = 64
save_iters = 1

# everything below here NEEDS to stay the same to load an extistng model
n_embed = 2560
n_head = 20
n_layer = 40
dropout = 0.2
vocab_size = 30000


#-------------------------------------------------------------
# DEFINE ESTIMATE LOSS FUNCTION
@torch.no_grad()
def estimate_loss(step):
    out = {}
    m.eval()
    trainval_sampler.set_epoch(step)
    val_sampler.set_epoch(step)
    t_it = iter(trainval_loader)
    v_it = iter(val_loader)
    running_losses = torch.zeros(2, device=device)
    for k in range(eval_iters):
        xt, yt = next(t_it)
        xv, yv = next(v_it)
        xt, yt = xt.to(device), yt.to(device)
        xv, yv = xv.to(device), yv.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, tloss = m(xt, yt)
            _, vloss = m(xv, yv)
        running_losses[0] += tloss
        running_losses[1] += vloss
    running_losses /= eval_iters
    dist.all_reduce(running_losses, op=dist.ReduceOp.SUM)
    running_losses /= world_size
    out['train'] = running_losses[0].item()
    out['val'] = running_losses[1].item()
    m.train()
    return out



#-------------------------------------------------------------
# CHECKPOINT FUNCTION
def save_checkpoint(step, epoch, block, model, optimizer, path):
    """saves the current model and optimizer state to the checkpoint path specified at the top
    Prints to confirm completion"""

    if hasattr(model, "_orig_mod"):
        model_to_save = model._orig_mod
    else:
        model_to_save = model

    with FSDP.state_dict_type(model_to_save, StateDictType.LOCAL_STATE_DICT):
        state = model_to_save.state_dict()

        clean_state = {}
        for key, value in state.items():
            new_key = key
            while True:
                if new_key.startswith("_orig_mod."):
                    new_key = new_key.replace("_orig_mod.", "", 1)
                elif new_key.startswith("_fsdp_wrapped_module."):
                    new_key = new_key.replace("_fsdp_wrapped_module.", "", 1)
                else:
                    break # Key is clean
            clean_state[new_key] = value

        checkpoint = {
            'model': clean_state,
            'optimizer': optimizer.state_dict(),
            'meta': {'step': step, 'epoch': epoch, 'block': block}
            }
        
        torch.save(checkpoint, f"{path}/rank{local_rank}.pt")
    print(f"Rank {local_rank} saved local shard")

    dist.barrier()
    del checkpoint, clean_state, state
    torch.cuda.empty_cache()

#-------------------------------------------------------------
if local_rank == 0:
    input("# INITIALIZE AND LOAD MODEL AND OPTIMIZER")

dist.barrier(device_ids=[local_rank])

m = GPTLanguageModel(vocab_size, n_embed, n_head, n_layer, dropout)

master_print(f"Total parameters {sum(p.numel() for p in m.parameters()):,}")

mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,)
my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e6)
m = FSDP(m,
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True 
)

#non_reentrant_wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
#apply_activation_checkpointing(m, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=lambda x: isinstance(x, Block))

optim = bnb.optim.PagedAdamW8bit(m.parameters(), lr=learning_rate)

if save:
    if local_rank == 0:
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    path = f"{CHECKPOINT_PATH}/rank{local_rank}.pt"
    dist.barrier()
    # check for an existing checkpoint and load if necessary
    if os.path.exists(path):
        print(f"Rank {local_rank} loading checkpoint: {path}...")
        checkpoint = torch.load(path, map_location="cpu")
        start_optim_step = checkpoint['meta']['step']
        optim_step = start_optim_step
        start_epoch = checkpoint['meta']['epoch']
        start_block = checkpoint['meta']['block']
        block = start_block

        if hasattr(m, "_orig_mod"):
            model_to_load = m._orig_mod
        else:
            model_to_load = m

        with FSDP.state_dict_type(model_to_load, StateDictType.LOCAL_STATE_DICT):
            missing, unexpected = model_to_load.load_state_dict(checkpoint['model'], strict=False)

            if len(missing) > 0:
                master_print(f"CRITICAL ERROR: The checkpoint is missing parameters!")
                master_print(f"Missing: {missing}")
                raise RuntimeError("Checkpoint load failed: Parameters are missing.")
            
            filtered_unexpected = [k for k in unexpected if "_flat_param" not in k]
            
            if len(filtered_unexpected) > 0:
                master_print(f"WARNING: Found unexpected keys that are NOT flat params:")
                master_print(f"{filtered_unexpected}")
            else:
                master_print("--> Checkpoint verification passed: No missing parameters.")
            
            optim.load_state_dict(checkpoint['optimizer'])

        del checkpoint, model_to_load
        torch.cuda.empty_cache()

        master_print(f"Loaded succesfuly from (step: {optim_step}, epoch: {start_epoch}, block: {start_block})")
    else:
        master_print(f"No checkpoint found at {path}.")
        master_print(f"New model will be training from (step: {optim_step}, epoch: {start_epoch}, block: {start_block})")

m = torch.compile(m)

gc.collect()

print(f"Rank {local_rank} Ready. Total Params: {sum(p.numel() for p in m.parameters()):,}")


#-------------------------------------------------------------
# Datasets and dataloading
class DistributedEvalSampler(DistributedSampler):
    def __init__(self, dataset, map_path, num_replicas=None, rank=None, seed=42, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.indices_np = np.memmap(map_path, dtype=np.int64, mode='r')
        self.indices_torch = torch.from_numpy(self.indices_np)
        
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        random_start = torch.randint(0, len(self.indices_torch) - (minibatch_size*eval_iters*self.num_replicas + 1), (1,), generator=g).item()
        chunk = self.indices_torch[random_start : random_start + minibatch_size*eval_iters*self.num_replicas]
        my_indices = chunk[self.rank :: self.num_replicas]
        
        yield from my_indices.tolist()

    def __len__(self):
        return minibatch_size*accumulation_steps
    
class DistributedTrainSampler(DistributedSampler):
    def __init__(self, dataset, map_path, start_pos, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.indices_np = np.memmap(map_path, dtype=np.int64, mode='r')
        self.start_block = start_pos
        self.indices_torch = torch.from_numpy(self.indices_np)

    def __iter__(self):
        remaining = self.indices_torch[self.start_block:]
        my_indices = remaining[self.rank::self.num_replicas]
        yield from my_indices.tolist()

    def __len__(self):
        remaining = len(self.indices_torch) - self.start_block
        return (remaining + self.num_replicas - 1) // self.num_replicas
        
class BinDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i):
        i = i * self.block_size
        chunk = self.data[i : i + self.block_size + 1]
        
        t = torch.from_numpy(chunk.astype(np.int64))
        
        x = t[:-1] # Input
        y = t[1:]  # Target (shifted right)
        return x, y
    
def get_loaders(data_path, batch_size, block_size, map_path, start_pos=0, estimate=False):
    dataset=BinDataset(data_path, block_size)

    if estimate:
        sampler = DistributedEvalSampler(dataset, map_path, world_size, local_rank)
    else:
        sampler = DistributedTrainSampler(dataset, map_path, start_pos, world_size, local_rank)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,      # More than 4 often causes overhead
        pin_memory=True,    # Fast transfer to GPU
        drop_last=True,      # Avoids partial batches that break FSDP shapes
        persistent_workers=True if estimate else False
    )
    
    return loader, sampler

train_loader, train_sampler = get_loaders(train_dataset_path, minibatch_size, block_size, train_map_paths[start_epoch], start_block)

val_loader, val_sampler = get_loaders(test_dataset_path, minibatch_size, block_size, val_map_path, estimate=True)
trainval_loader, trainval_sampler = get_loaders(train_dataset_path, minibatch_size, block_size, train_map_paths[start_epoch], estimate=True)

#-------------------------------------------------------------
dist.barrier()
if local_rank == 0:
    print()
    input("# TRAINING LOOP")

dist.barrier(device_ids=[local_rank])

master_print(f"{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")
losses = estimate_loss(optim_step)    # estimate a base loss before training session
print()
master_print(f"Starting loss: (step: {optim_step}, epoch: {start_epoch}, block: {start_block}), Train Loss: {losses['train']:.3f}, Val Loss: {losses['val']:.3f}")
master_print(f"{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")

if save and int(os.environ.get("LOCAL_RANK", 0)) == 0:
    # if never trained log the pre optim loss in a new csv
    if optim_step == 0:
        print(f"Initializing new data collection file at: {LOG_FILE}")
        with open(LOG_FILE, 'a') as f:
            f.write(f"step,epoch,block,train_loss,val_loss\n{optim_step:05d},{start_epoch:03d},{start_block:05d},{losses['train']:.3f},{losses['val']:.3f}\n")

dist.barrier()

# Dictionary to hold total accumulated time and count
times_tracker = defaultdict(lambda: {'time': 0.0, 'count': 0})

train_start_event = torch.cuda.Event(enable_timing=True)
train_end_event = torch.cuda.Event(enable_timing=True)


for epoch in range(start_epoch, max_epochs+1):
    if epoch != start_epoch:
        start_block = 0
        train_loader, train_sampler = get_loaders(train_dataset_path, minibatch_size, block_size, train_map_paths[epoch], start_block)
        
    optim.zero_grad(set_to_none=True)
    train_start_event.record()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = m(x, y)
            loss = loss / accumulation_steps

        block += minibatch_size*world_size
        
        if (batch_idx + 1) % accumulation_steps != 0:
            with m.no_sync():
                loss.backward()
        else:
            loss.backward()
            optim.step()
            optim_step += 1
            optim.zero_grad(set_to_none=True)

            if (optim_step - start_optim_step) % save_iters == 0:
                train_end_event.record()
                torch.cuda.synchronize()
                # elapsed_time returns milliseconds, so divide by 1000.0
                elapsed_time_sec = train_start_event.elapsed_time(train_end_event) / 1000.0
                times_tracker['train']['time'] += elapsed_time_sec
                times_tracker['train']['count'] += save_iters
                
                tic = time.perf_counter()
                losses = estimate_loss(optim_step)
                torch.cuda.synchronize()
                times_tracker['estimate']['time'] += (time.perf_counter() - tic)
                times_tracker['estimate']['count'] += 1
                
                master_print(f"\n(step: {optim_step}, epoch: {epoch}, block: {block}), Train Loss: {losses['train']:.3f}, Val Loss: {losses['val']:.3f}")
                tic = time.perf_counter()
                
                if save:
                    save_checkpoint(optim_step, epoch, block, m, optim, CHECKPOINT_PATH)
            
                    # write a new line in our data csv
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        with open(LOG_FILE, 'a') as f:
                            f.write(f"{optim_step:05d},{epoch:03d},{block:05d},{losses['train']:.3f},{losses['val']:.3f}\n")
                        print(f"Step Documented")

                times_tracker['save']['time'] += (time.perf_counter() - tic)
                times_tracker['save']['count'] += 1

                dist.barrier(device_ids=[local_rank])
                
                master_print(f"{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")
                master_print("")
                master_print(f"Total time train: {times_tracker['train']['time'] /60/60:.3f}hr")
                master_print(f"Total time estim: {times_tracker['estimate']['time'] /60/60:.3f}hr")
                master_print(f"Total time Check: {times_tracker['save']['time'] /60/60:.3f}hr")
                master_print(f"Average time per Optimizer step: {times_tracker['train']['time'] / times_tracker['train']['count']:.3f}sec")
                master_print(f"Average time {save_iters} Optimizr steps: {times_tracker['train']['time'] / (times_tracker['train']['count'] / save_iters)/60:.3f}min")
                master_print(f"Average time per Estimate  Loss: {times_tracker['estimate']['time'] / times_tracker['estimate']['count']:.3f}sec")
                master_print(f"Average time per Chckpoint Save: {times_tracker['save']['time'] / times_tracker['save']['count']:.3f}sec")
                train_start_event.record()