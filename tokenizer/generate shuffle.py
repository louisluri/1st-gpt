import numpy as np
import os

def generate_shuffle_map(output_path, total_tokens, block_size, seed=42):
    # 1. Calculate how many chunks (slots) we have
    num_chunks = (total_tokens - 1) // block_size
    print(f"Generating map for {num_chunks} chunks...")

    # 2. Create a memmap file to store the shuffled indices
    # 'w+' creates a new file; 'int64' ensures we can hold large indices
    m = np.memmap(output_path, dtype=np.int64, mode='w+', shape=(num_chunks,))

    # 3. Fill it with sequential numbers first
    m[:] = np.arange(num_chunks)

    # 4. Shuffle it using a seeded generator
    # This is the "Magic" part: we shuffle the memmap directly on disk
    rng = np.random.default_rng(seed)
    rng.shuffle(m)

    # 5. Flush to disk and close
    m.flush()
    print(f"Shuffle map saved to {output_path}")

# Usage
generate_shuffle_map('gpt_dt/openwebtext1/val_shuffle_map0-128block.bin', 888_249_069, 128)

#data = np.memmap('gpt_dt/openwebtext1/test_data.bin', dtype=np.uint16, mode='r')
#print(f"Total tokens (memmap): {len(data):,}")