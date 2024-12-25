import math
import torch
import re
import torch.nn as nn
import tqdm

from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

def seed_everything(seed=42):
  import random, torch
  import numpy as np
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

# linear only
LINEAR_LAYER_ONLY_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|proj_(in|out))$"

# ALL \ time embed
# DEFAULT_LAYER_REGEX = "(down|mid|up)_blocks?.*(to_(q|k|v|out.0)|net\.(0\.proj|2)|conv(\d+|_shortcut)?|proj_(in|out))$"

def get_linear_layers(pipe, refilter=None, min_channels=16, layer_to_find=nn.Linear):
  if refilter is None:
    refilter = LINEAR_LAYER_ONLY_REGEX

  def layer_filter_fn(layer: nn.Module, layer_name: str) -> bool:
    # if isinstance(layer, (nn.Conv2d)):
    #   if min(layer.in_channels, layer.out_channels) < min_channels:
    #     return
    if isinstance(layer, layer_to_find):
      if min(layer.in_features, layer.out_features) < min_channels:
        return
    else:
      return
    return re.search(refilter, layer_name)

  # collect groups
  down_group = []
  # collect from down blocks
  for i, block in enumerate(pipe.unet.down_blocks):
    group = {}
    for module_name, module in block.named_modules():
      full_module_name = f"down_blocks.{i}.{module_name}"
      if layer_filter_fn(module, full_module_name):
        group[full_module_name] = module
    down_group.append(list(group.items()))

  # collect from mid block
  group = {}
  block = pipe.unet.mid_block
  for module_name, module in block.named_modules():
    full_module_name = f"mid_block.{module_name}"
    if layer_filter_fn(module, full_module_name):
      group['full_module_name'] = module
  mid_group = [list(group.items())]

  up_group = []
  # collect from up blocks
  for i, block in enumerate(pipe.unet.up_blocks):
    group = {}
    for module_name, module in block.named_modules():
      full_module_name = f"up_blocks.{i}.{module_name}"
      if layer_filter_fn(module, full_module_name):
        group['full_module_name'] = module
    up_group.append(list(group.items()))
  all_layers = down_group+mid_group+up_group
  all_layers = [i[1] for j in all_layers for i in j]
  return all_layers


def q_unet(pipe, nbits=4):
  all_layers = get_linear_layers(pipe)
  quant_config = BaseQuantizeConfig(nbits=nbits, group_size=64)

  all_quantized = []
  seed_everything()
  for layer in all_layers:
    hqq_layer = HQQLinear(layer, #torch.nn.Linear or None 
                          quant_config=quant_config, #quantization configuration
                          compute_dtype=torch.float16, #compute dtype
                          device='cuda', #cuda device
                          initialize=True, #Use False to quantize later
                          del_orig=True #if True, delete the original layer
                          )
    all_quantized.append(hqq_layer)

  for orig_layer, quantized_layer in zip(all_layers, all_quantized):
    found_original = False
    modules_to_update = []
    for submodule in pipe.unet.modules():
      for child_name, child_module in submodule.named_children():
        if child_module is orig_layer:
          modules_to_update.append((submodule, child_name))
          found_original = True
    assert found_original, f"could not find {orig_layer}"

    for submodule, child_name in modules_to_update:
      setattr(submodule, child_name, quantized_layer)
  return all_quantized


def generate_hadamard(n):
  """
  Generate an n x n Hadamard matrix using Sylvester's construction.
  n must be a power of two.
  """
  H = torch.tensor([[1.]], dtype=torch.float16)
  while H.size(0) < n:
    H = torch.cat([
      torch.cat([H, H], dim=1),
      torch.cat([H, -H], dim=1)
    ], dim=0)
  return H

def random_hadamard_transform_inplace(W, block_size=128, seed=42):
  """
  Apply a random Hadamard transform in-place to the 'weight' of the given nn.Linear layer.
  Assumes in_features is divisible by block_size and block_size is a power-of-two.
  The transform is (1/sqrt(block_size)) * ( W_block * D ) * H for each block,
  where D is the sign-flip diagonal matrix, and H is the Hadamard matrix.
  """
  with torch.no_grad():
    if isinstance(W, torch.nn.modules.linear.Linear):
      W = W.weight.data # shape: [out_features, in_features]
    in_features = W.shape[1]
    if in_features % block_size != 0:
      raise ValueError("in_features must be divisible by block_size.")

    # Precompute the Hadamard matrix (not normalized)
    H = generate_hadamard(block_size).to(W.device)

    # Number of blocks
    num_blocks = in_features // block_size

    # We will reuse the same sign-flip vector for each block
    # (Or you could make it block-dependent if desired.)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    sign_vector = (torch.randint(0, 2, (block_size,), generator=gen, device='cpu') * 2.0 - 1.0).to(W.device)

    # Transform each block
    for i in range(num_blocks):
      start = i * block_size
      end = (i+1) * block_size

      # Extract block: shape [out_features, block_size]
      W_block = W[:, start:end]

      # 1) Randomly flip signs (D matrix)
      W_block *= sign_vector.unsqueeze(0)

      # 2) Apply the Hadamard transform W_block @ H
      W_block = W_block @ H

      # Write back
      W[:, start:end].copy_(W_block)

    # 3) Scale the entire weight matrix by 1 / sqrt(block_size)
    W /= math.sqrt(block_size)


def inverse_random_hadamard_transform_inplace(W, block_size=128, seed=42):
  """
  Invert the random Hadamard transform in-place for the given nn.Linear layer,
  assuming the same seed and block_size that was used during forward transform.
  The inverse transform is sqrt(block_size) * W_block * H^-1 * D,
  with H^-1 = (1/block_size) * H (for the standard Sylvester Hadamard) and D = D^-1.
  """
  with torch.no_grad():
    if isinstance(W, torch.nn.modules.linear.Linear):
      W = W.weight.data # shape: [out_features, in_features]
    in_features = W.shape[1]
    if in_features % block_size != 0:
      raise ValueError("in_features must be divisible by block_size.")

    # Precompute the Hadamard matrix (not normalized) and its inverse
    # Since H·H = block_size * I for the Sylvester Hadamard, H^-1 = (1/block_size) H.
    H = generate_hadamard(block_size).to(W.device)
    H_inv = H / block_size

    # Number of blocks
    num_blocks = in_features // block_size

    # Same random sign flips as before
    gen = torch.Generator(device='cpu').manual_seed(seed)
    sign_vector = (torch.randint(0, 2, (block_size,), generator=gen, device='cpu') * 2.0 - 1.0).to(W.device)

    # 1) Undo the global scale by multiplying entire matrix by sqrt(block_size)
    W *= math.sqrt(block_size)

    # 2) For each block, apply the inverse steps
    for i in range(num_blocks):
      start = i * block_size
      end = (i+1) * block_size

      # Extract block
      W_block = W[:, start:end]

      # Apply H_inv
      W_block = W_block @ H_inv

      # Apply the same sign flips again
      W_block *= sign_vector.unsqueeze(0)

      # Write back
      W[:, start:end].copy_(W_block)


def dequantize_unet(pipe, all_layers, block_size=128, seed=42):
  """
  Replace each HQQLinear in `all_layers` by a standard nn.Linear having:
  - the dequantized float weights
  - the (optionally) dequantized float bias
  - inverse Hadamard transform applied to the float weights
  """
  new_layers = []
  for layer in tqdm.tqdm(all_layers, "dequantizing & inv. transform"):
    # 1) Obtain float (dequantized) weights/bias:
    weight_f16 = layer.dequantize().clone()
    bias_f16 = layer.bias if layer.bias is not None else None

    # 2) Apply inverse Hadamard transform in place:
    inverse_random_hadamard_transform_inplace(weight_f16, block_size=block_size, seed=seed)

    # 3) Create a new nn.Linear with the same shape:
    out_features, in_features = weight_f16.shape
    new_linear = nn.Linear(in_features, out_features, bias=(bias_f16 is not None))
    new_linear.weight = nn.Parameter(weight_f16)
    if bias_f16 is not None:
      new_linear.bias = nn.Parameter(bias_f16)
    new_layers.append(new_linear)

  # 4) Replace HQQLinear modules in the pipe.unet with the new nn.Linear modules:
  for orig_layer, new_linear in zip(all_layers, new_layers):
    found_original = False
    modules_to_update = []
    for submodule in pipe.unet.modules():
      for child_name, child_module in submodule.named_children():
        if child_module is orig_layer:
          modules_to_update.append((submodule, child_name))
          found_original = True
    assert found_original, f"Could not find the original layer {orig_layer} in pipe.unet."

    for submodule, child_name in modules_to_update:
      setattr(submodule, child_name, new_linear)

  return new_layers


