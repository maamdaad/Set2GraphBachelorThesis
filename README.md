# Set2GraphPaper
## Installation

    module load cuda
    
    cd SetToGraphPaper
    python download_jets_data.py

    conda create -n s2g_env -c pytorch pytorch=1.5 cudatoolkit=10.2 torchvision  
    conda activate s2g_env

    CUDA=cu102
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
    pip install torch-spline-conv
    pip install torch-geometric

##JobScript

    #!/usr/local_rwth/bin/zsh
    #SBATCH --mem-per-cpu=20480M
    #SBATCH --gres=gpu:volta:2
    #SBATCH --job-name=S2G_Paper
    #SBATCH --time=01:00:00
    #SBATCH --output=output.%J.txt
    ### begin of executable commands
    module load cuda
    cd /home/rq388478/Bachelorarbeit/SetToGraphPaper
    source /home/rq388478/miniconda3/bin/activate s2g_env
    python main_scripts/main_jets.py --load_model 1

