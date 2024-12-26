import yaml
import subprocess
import os

ROOT = '/home/mdnikolaev/philurame'
PROJ_ROOT = os.path.join(ROOT, 'Experiments/QuatizationClip')

from itertools import product
for nbits in [16,8,6,5,4,3,2,1]:
  method =  f'{nbits}'
  outdir = os.path.join(ROOT, f'_runs/Q')
  os.makedirs(outdir, exist_ok=True)

  # Build the script content
  script_content = f"""\
#!/bin/bash --login
#SBATCH --job-name={method}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --constraint="[type_a|type_b|type_c|type_e]"
#SBATCH --output={outdir}/{method}-%j.log

module load Python/Anaconda_v03.2023

conda deactivate
conda activate philurame_venv

python3 /home/mdnikolaev/philurame/Experiments/QuantizationClip/get_clips.py \\
--nbits {nbits}
"""

  # Submit the script content via sbatch
  command = ['sbatch']
  subprocess.run(command, input=script_content, text=True)