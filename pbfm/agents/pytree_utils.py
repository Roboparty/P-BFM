# Custom utilities for handling PyTrees of torch tensors

from torch.utils._pytree import tree_flatten


def tree_get_batch_size(pytree):
    tensors, _ = tree_flatten(pytree)
    batch_sizes = [t.shape[0] for t in tensors]
    assert all(bs == batch_sizes[0] for bs in batch_sizes), f"All tensors must have the same batch size {batch_sizes[0]}, got {batch_sizes}"
    return batch_sizes[0]
