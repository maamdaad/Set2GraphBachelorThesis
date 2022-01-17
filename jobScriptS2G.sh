#!/usr/local_rwth/bin/zsh
 
#SBATCH --mem-per-cpu=20480M  
#SBATCH --gres=gpu:volta:2 
#SBATCH --job-name=S2G_Paper
#SBATCH --time=01:00:00
#SBATCH --output=output.%J.txt
 
### begin of executable commands
module load cuda
cd /work/directory/
source /path/to/python/env/bin/activate s2g_env
python main_scripts/main_jets.py --load_model 1 
