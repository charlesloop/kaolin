# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import torch


def manual_seed(torch_seed, random_seed=None, numpy_seed=None):
    """Set the seed for random and torch modules.

    Args:
        torch_seed (int): The desired seed for torch module.
        random_seed (int): The desired seed for random module. Default: torch_seed value.
        numpy_seed (int): The desired seed for numpy module. Default: torch_seed value.
    """
    if random_seed is None:
        random_seed = torch_seed
    if numpy_seed is None:
        numpy_seed = torch_seed
    random.seed(random_seed)
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)

def set_state(torch_state, random_state, numpy_state):
    """Set the generator states for generating random numbers.

    Mostly used in pair with :func:`get_state`

    Args:
        torch_state (torch.ByteTensor): the state of torch module.
        random_state (tuple): the state of random module.
        numpy_state (tuple): the state of numpy module.

    Example:
        >>> torch_state, random_state, numpy_state = get_state()
        >>> s = torch.randn((1, 3))
        >>> set_state(torch_state, random_state, numpy_state)
    """
    torch.set_rng_state(torch_state)
    random.setstate(random_state)
    np.random.set_state(numpy_state)

def get_state():
    """Returns the generator states for generating random numbers.

    Mostly used in pair with :func:`set_state`
    pytest --doctest-modules kaolin/
    - https://pytorch.org/docs/stable/generated/torch.get_rng_state.html#torch.get_rng_state
    - https://docs.python.org/3/library/random.html#random.getstate
    - https://numpy.org/doc/stable/reference/random/generated/numpy.random.set_state.html#numpy.random.set_state

    Returns:
       (torch.ByteTensor, tuple, tuple):
           the states for the corresponding modules (torch, random, numpy).

    Example:
        >>> torch_state, random_state, numpy_state = get_state()
        >>> s = torch.randn((1, 3))
        >>> set_state(torch_state, random_state, numpy_state)
    """
    return torch.get_rng_state(), random.getstate(), np.random.get_state()

def random_shape_per_tensor(batch_size, min_shape=None, max_shape=None):
    """Generate random :attr:`shape_per_tensor`.

    Args:
        min_shape (list, tuple or torch.LongTensor): minimum values for each dimension of generated shapes.
            Default: 1 for each dimensions.
        max_shape (list, tuple or torch.LongTensor): maximu values for each dimension of generated shapes.

    Return:
        (torch.LongTensor): A shape_per_tensor (2D).

    Example:
        >>> _ = torch.random.manual_seed(1)
        >>> random_shape_per_tensor(3, min_shape=(4, 4), max_shape=(10, 10))
        tensor([[ 4,  7],
                [ 7,  7],
                [ 8, 10]])
    """
    if min_shape is None:
        min_shape = [1] * len(max_shape)
    output = torch.cat([torch.randint(low_dim, high_dim + 1, size=(batch_size, 1))
                        for low_dim, high_dim in zip(min_shape, max_shape)], dim=1)
    return output

def random_tensor(low, high, shape, dtype=torch.float, device='cpu'):
    """Generate a random tensor.

    Args:
        low (float): the lowest value to be drawn from the distribution.
        high (float): the highest value to be drawn from the distribution.
        shape (list, tuple or torch.LongTensor): the desired output shape.
        dtype (torch.dtype): the desired output dtype.
        device (torch.device): the desired output device.

    Return:
        (torch.Tensor): a random generated tensor.

    Example:
        >>> _ = torch.random.manual_seed(1)
        >>> random_tensor(4., 5., (3, 3), dtype=torch.float, device='cpu')
        tensor([[4.7576, 4.2793, 4.4031],
                [4.7347, 4.0293, 4.7999],
                [4.3971, 4.7544, 4.5695]])
    """
    if dtype in (torch.half, torch.float, torch.double):
        output = torch.rand(shape, dtype=dtype, device=device)
        if (low != 0.) or (high != 1.):
            output = output * (high - low) + low
    elif dtype == torch.bool:
        assert (low is None) or (low == 0)
        assert (high is None) or (high == 1)
        output = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        output = torch.randint(low, high + 1, size=shape, dtype=dtype, device=device)
    return output
