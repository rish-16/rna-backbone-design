# basic ML
# change pytorch-cuda version according to your system's specifications
conda install pytorch=2.1.2 torchvision torchaudio pytorch-cuda==11.8 -c pytorch -c nvidia
pip install lightning==2.0.7
pip install hydra-core==1.3.2

# pyg (optional to include pytorch-sparse and pytorch-spline-conv)
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

# molecular preprocessing
conda install mdanalysis MDAnalysisTests -c conda-forge -y
conda install biopandas biopython -c conda-forge -y
conda install openbabel -c conda-forge -y
pip install rdkit
pip install mdtraj
pip install graphein

# misc
pip install wandb hydra-colorlog rootutils rich matplotlib networkx gputil omegaconf beartype jaxtyping dm-tree tmtools POT iminuit tmscoring