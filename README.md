# SSSNET
SSSNET: Semi-Supervised Signed Network Clustering (accepted by SDM2022)

For details, please read [our paper](https://arxiv.org/pdf/2110.06623.pdf).

## Environment Setup
### Overview
<!-- The underlying project environment composes of following componenets: -->
The project has been tested on the following environment specification:
1. Ubuntu 18.04.5 LTS (Other x86_64 based Linux distributions should also be fine, such as Fedora 32)
2. Nvidia Graphic Card (NVIDIA GeForce RTX 2080 with driver version 440.36, and NVIDIA RTX 8000) and CPU (Intel Core i7-10700 CPU @ 2.90GHz)
3. Python 3.6.13 (and Python 3.6.12)
4. CUDA 10.2 (and CUDA 9.2)
5. Pytorch 1.8.0 (built against CUDA 10.2) and Python 1.6.0 (built against CUDA 9.2)
6. Other libraries and python packages (See below)

You should handle (1),(2) yourself. For (3), (4), (5) and (6), see following methods.

### Installation Method 1 (Using Installation Script)
<!-- There are two options for you to install the requirements, you could either follow the next subsection to update your installations, or follow the instructions below in this subsection. -->

<!-- For the latter: -->



<!-- We place those python packages that can be easily installed with one-line command in the requirement file for `pip` (`requirements_pip.txt`). For all other python packages, which are not so well maintained by [PyPI](https://pypi.org/), and all C/C++ libraries, we place in the conda requirement file (`requirements_conda.txt`). Therefore, you need to run both conda and pip to get necessary dependencies. -->

We provide two examples of environmental setup, one with CUDA 10.2 and GPU, the other with CPU.

Following steps assume you've done with (1) and (2).
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Both Miniconda and Anaconda are OK.

2. Run the following bash script under SSSNET's root directory.
```bash
./create_conda_env.sh
```

### Installation Method 2 (.yml files)
We provide two examples of envionmental setup, one with CUDA 10.2 and GPU, the other with CPU.

Following steps assume you've done with (1) and (2).
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Both Miniconda and Anaconda are OK.

2. Create an environment and install python packages (GPU):
```
conda env create -f environment_GPU.yml
```

3. Create an environment and install python packages (CPU):
```
conda env create -f environment_CPU.yml
```


### Installation Method 3 (Manually Install)
The codebase is implemented in Python 3.6.12. package versions used for development are just below.
```
networkx           2.5
tqdm               4.50.2
numpy              1.19.2
pandas             1.1.4
texttable          1.6.3
latextable         0.1.1
scipy              1.5.4
argparse           1.1.0
sklearn            0.23.2
torch              1.8.1
torch-scatter      2.0.5
torch-geometric    1.6.3 (follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
matplotlib         3.3.4 (for generating plots and results)
SigNet         (for comparison methods, can get from the command: pip install git+https://github.com/alan-turing-institute/SigNet.git)
```

### Execution checks
When installation is done, you could check you enviroment via:
```
bash setup_test.sh
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use [GNU parallel](https://www.gnu.org/software/parallel/), can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores processed data sets for node clustering.

- ./src/ stores files to train various models, utils and metrics.

- ./result_arrays/ stores results for different data sets. Each data set has a separate subfolder.

- ./result_anlysis/ stores notebooks for generating result plots or tables.

- ./logs/ stores trained models and logs, as well as predicted clusters (optional). When you are in debug mode (see below), your logs will be stored in ./debug_logs/ folder.

## Options
<p align="justify">
SSSNET provides the following command line arguments, which can be viewed in the ./src/param_parser.py and ./src/link_sign_param_parser.py.
</p>

### Synthetic data options:
See file ./src/param_parser.py.

```
  --p                     FLOAT         Probability of the existence of a link.                 Default is 0.02. 
  --eta                   FLOAT         Probability of flipping the sign of each edge.          Default is 0.1.
  --N                     INT           (Expected) Number of nodes in an SSBM.                  Default is 1000.
  --K                     INT           Number of blocks in an SSBM.                            Default is 3.
  --total_n               INT           Total number of nodes in the polarized network.         Default is 1050.
  --num_com               INT           Number of polarized communities (SSBMs).                Default is 2.
```

### Major model options:
See file ./src/param_parser.py.

```
  --epochs                INT         Number of SSSNET (maximum) training epochs.               Default is 300. 
  --early_stopping        INT         Number of SSSNET early stopping epochs.                   Default is 100. 
  --num_trials            INT         Number of trials to generate results.                     Default is 10.
  --seed_ratio            FLOAT       Ratio in the training set of each cluster 
                                                        to serve as seed nodes.                 Default is 0.1.
  --loss_ratio            FLOAT       Ratio of loss_pbnc to loss_pbrc. -1 means only loss_pbnc. Default is -1.0.
  --supervised_loss_ratio FLOAT       Ratio of factor of supervised loss part to
                                      self-supervised loss part.                                Default is 50.
  --triplet_loss_ratio    FLOAT       Ratio of triplet loss to cross entropy loss in 
                                      supervised loss part.                                     Default is 0.1.
  --tau                   FLOAT       Regularization parameter when adding self-loops to the positive 
                                      part of the adjacency matrix, i.e. A -> A + tau * I,
                                      where I is the identity matrix.                           Default is 0.5.
  --hop                   INT         Number of hops to consider for the random walk.           Default is 2.
  --samples               INT         Number of samples in triplet loss.                        Default is 10000.
  --train_ratio           FLOAT       Training ratio.                                           Default is 0.8.  
  --test_ratio            FLOAT       Test ratio.                                               Default is 0.1.
  --lr                    FLOAT       Initial learning rate.                                    Default is 0.01.  
  --weight_decay          FLOAT       Weight decay (L2 loss on parameters).                     Default is 5^-4. 
  --dropout               FLOAT       Dropout rate (1 - keep probability).                      Default is 0.5.
  --hidden                INT         Number of hidden units.                                   Default is 32. 
  --seed                  INT         Random seed.                                              Default is 31.
  --no-cuda               BOOL        Disables CUDA training.                                   Default is False.
  --debug, -D             BOOL        Debug with minimal training setting, not to get results.  Default is False.
  --directed              BOOL        Directed input graph.                                     Default is False.
  --no_validation         BOOL        Whether to disable validation and early stopping
                                      during traing.                                            Default is False.
  --regenerate_data       BOOL        Whether to force creation of data splits.                 Default is False.
  --load_only             BOOL        Whether not to store generated data.                      Default is False.
  --dense                 BOOL        Whether not to use torch sparse.                          Default is False.
  -AllTrain, -All         BOOL        Whether to use all data to do gradient descent.           Default is False.
  --SavePred, -SP         BOOL        Whether to save predicted labels.                         Default is False.
  --dataset               STR         Data set to consider.                                     Default is 'SSBM/'.
  --all_methods           LST         Methods to use to generate results.                       Default is ['spectral','SSSNET'].
  --feature_options       LST         Features to use for SSSNET. 
                                      Can choose from ['A_reg','L','given','None'].            Default is ['A_reg'].
```

## Reproduce results
First, get into the ./execution/ folder:
```
cd execution
```
To reproduce SSBM results.
```
bash SSBM.sh
```
To reproduce results on polarized SSBMs.
```
bash polarized.sh
```
To reproduce results of node clustering on real data.
```
bash real.sh
```

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files.

You can also use CPU for training if you add ``--no-duca", or GPU if you delete this.

## Direct execution with training files

First, get into the ./src/ folder:
```
cd src
```

Then, below are various options to try:

Creating an SSSNET model for SSBM of the default setting.
```
python ./train.py
```
Creating an SSSNET model for polarized SSBMs with 5000 nodes, N=500.
```
python ./train.py --dataset polarized --total_n 5000 --N 500
```
Creating a model for S&P1500 data set with some custom learning rate and epoch number.
```
python ./train.py --dataset SP1500 --lr 0.001 --epochs 300
```
Creating a model for Wiki-Rfa data set (directed) with specific number of trials and use CPU.
```
python ./train.py --dataset wikirfa --directed --no-cuda --num_trials 5
```

## Note
- When no ground-truth exists, the labels loaded for training/testing are not really meaningful. They simply provides some relative Adjusted Rand Index (ARI) for the model's predicted clustering to some fitted/dummy clustering. The codebase loads these fitted/dummy labels and prints out ARIs for completeness instead of evaluation purpose.

- Other versions of the code. ./src/PyG_models.py provides a version of SSSNET with pytorch geometric message passing implementation, and ./src/PyG_train.py runs this model for SSSNET. Note that this implementation gives almost the same results.

--------------------------------------------------------------------------------