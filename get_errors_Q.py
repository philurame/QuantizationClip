import sys, torch, pickle, click

sys.path.append('/home/mdnikolaev/philurame/Experiments/EpsilonApprox')
from q_utils import *

sys.path.append('/home/mdnikolaev/philurame/SDXL_METRICS')
from main import load_pipe, load_data
from lib.generate_decode import generate, decode_vae
from lib.metric import metrics_registry


@click.command()
@click.option('--nbits', type=int, required=True, help='nbits')
@click.option('--batch', type=int, required=True, help='nbits')
def main(nbits, batch):

  pipe = load_pipe(solver='DDIM', scheduler='LINEAR', cacher_quantizer='NONE', is_optimize=False, add_vae=True)

  all_layers = get_linear_layers(pipe)
  orig_weights = [i.weight.clone() for i in all_layers]

  # apply hadamard transform to all linear layers
  for layer in all_layers:
    random_hadamard_transform_inplace(layer)

  # quantize using HQQ
  all_quantized_layers = q_unet(pipe, nbits=nbits)

  # dequantize
  new_layers = dequantize_unet(pipe, all_quantized_layers)

  # get L2 errors
  errors = [torch.norm(orig_weights[i]-new_layers[i].weight).item() for i in range(len(orig_weights))]

  # generate 100 images for clip:
  coco_anns = load_data('/home/mdnikolaev/philurame/SDXL_METRICS/DATA', 'COCO', 2100).anns[batch*100:(batch+1)*100]
  fake_latents = generate(pipe, coco_anns, nfe=50)

  # log
  with open(f'/home/mdnikolaev/philurame/Experiments/QuantizationClip/RES/{nbits}_{batch}.pkl', 'wb') as f:
    pickle.dump({'errors': errors,'fake_latents': fake_latents}, f)
  

if __name__ == '__main__':
  main()
