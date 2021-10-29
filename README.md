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


