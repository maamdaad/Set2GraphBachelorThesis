# Set2GraphPaper

MiniConda (https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

## Installation

    module load cuda
    
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

## Running

Training and Testing:

    python main_scripts/main_jets.py --method=lin2 [--vram_clear_time=1]

Testing:

    python main_scripts/main_jets.py --load_model=1 --vram_clear_time=1 --model_path="../experiments/jets_results/jets_20211102_234945_0/exp_model.pt"


