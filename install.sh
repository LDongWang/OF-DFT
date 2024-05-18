conda create -y -n realdft python=3.9
source activate realdft
pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
conda install pandas numpy scipy matplotlib seaborn -y
pip install jupyter pyscf e3nn jupytext jupyter_contrib_nbextensions
pip install --upgrade notebook==6.4.12
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
