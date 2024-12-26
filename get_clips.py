import torch, pickle, sys, tqdm, click
import numpy as np
import pandas as pd

sys.path.append('/home/mdnikolaev/philurame/Experiments/QuantizationClip')
from q_utils import *

sys.path.append('/home/mdnikolaev/philurame/SDXL_METRICS')
from main import load_pipe, load_data
from lib.generate_decode import decode_vae
from lib.metric import metrics_registry

@click.command()
@click.option('--nbits', type=int, required=True, help='nbits')
def main(nbits):

  coco_data = load_data('/home/mdnikolaev/philurame/SDXL_METRICS/DATA', 'COCO', 2100)
  pipe      = load_pipe(solver='DDIM', scheduler='LINEAR', cacher_quantizer='NONE', is_optimize=False, add_vae=True)

  w_norms_16  = [torch.norm(W.weight).item() for W in get_linear_layers(pipe)]

  df = pd.DataFrame(columns=["batch", "errors", "clip"])
  for batch in tqdm.tqdm(range(20)):
    with open(f'/home/mdnikolaev/philurame/Experiments/QuantizationClip/RES/{nbits}_{batch}.pkl', 'rb') as f:
      errors_latents = pickle.load(f)
      errors  = errors_latents['errors']
      latents = errors_latents['fake_latents']
      real_anns = coco_data.anns[(batch*100):(batch+1)*100]
        
    # decode images
    gen_imgs = decode_vae(pipe, latents)

    # calc CLIP
    clip = metrics_registry['CLIP'](fake_imgs = gen_imgs, real_anns = real_anns)()

    df_temp = pd.DataFrame({
      "clip":  [clip]*len(errors),
      "batch": [batch]*len(errors),
      "errors": np.array(errors)**2 / np.array(w_norms_16)**2
    })
    df = pd.concat([df, df_temp], ignore_index=True)

  df.to_csv(f'/home/mdnikolaev/philurame/Experiments/QuantizationClip/RES/CLIPS[{nbits}].csv')

  latents  = torch.load('/home/mdnikolaev/philurame/SDXL_METRICS/DATA/NONE/COCO_50/DDIM_LINEAR.pt')[:1000]
  gen_imgs = decode_vae(pipe, latents)
  clip = metrics_registry['CLIP'](fake_imgs = gen_imgs, real_anns = coco_data.anns[:1000])()
  print(f'float16 CLIP = {clip}')
  sys.stdout.flush()

if __name__ == '__main__':
  main()