import yaml
import subprocess
import os

ROOT = '/home/mdnikolaev/philurame'
PROJ_ROOT = os.path.join(ROOT, 'Experiments/QuatizationClip')

from itertools import product
for nbits, batch in product([8,6,5,4,3,2,1], [1,2,3,4,5,6,7,8,9,10]):
  method =  f'{nbits}'
  outdir = os.path.join(ROOT, f'_runs/Q')
  os.makedirs(outdir, exist_ok=True)

  # Build the script content
  script_content = f"""\
#!/bin/bash --login
#SBATCH --job-name={method}_{batch}
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --constraint="[type_a|type_b|type_c|type_e]"
#SBATCH --output={outdir}/{method}_{batch}-%j.log

module load Python/Anaconda_v03.2023

conda deactivate
conda activate philurame_venv

python3 /home/mdnikolaev/philurame/Experiments/QuantizationClip/get_errors_Q.py \\
--nbits {nbits} \\
--batch {batch}
"""

  # Submit the script content via sbatch
  command = ['sbatch']
  subprocess.run(command, input=script_content, text=True)