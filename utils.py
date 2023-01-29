from jax.numpy import broadcast_to
from jax.tree_util import tree_map
from jax.nn import tanh
from typing import Optional


def add_batch(nest, batch_size: Optional[int]):
    broadcast = lambda x: broadcast_to(x, (batch_size,) + x.shape)
    return tree_map(broadcast, nest)

def lecun_tanh(x):
    return 1.7159 * tanh(x * 0.666)

