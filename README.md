# Set2GraphPaper

Ich habe MiniConda auf dem HPC installiert, schau mal (https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh): 

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

## Ausführen

Zum Trainieren und Testen:

    python main_scripts/main_jets.py --method=lin2

Zum Testen:

    python main_scripts/main_jets.py --load_model 1

Musst dann aber in main_jets.py in der `main()` den Pfad zu einem vorherigen Modell anpassen.

`main_jets_load_model.py` ist nicht wichtig, habe ich nur zum Debuggen benutzt.
