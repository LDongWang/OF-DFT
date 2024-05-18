**Overcoming the Barrier of Orbital-Free Density Functional Theory in Molecular Systems Using Deep Learning**
========

# System Requirements

## Hardware Requirements
This code should run on most modern computers, and a Nvidia GPU is highly recommended to take full advantage of M-OFDFT.

## Software Requirements

### OS Requirements
The code has been tested on Ubuntu 20.04.6 LTS. Other OS supported by PyTorch should also be fine.

### Python Dependencies
This code mainly depends on these Python packages:
```bash
- PyTorch (1.9.1)
- PyG (1.7.2)
- pyscf (2.3.0)
- pandas (2.0.3)
- numpy (1.26.0)
- scipy (1.11.3)
- e3nn (0.5.1)
- matplotlib (3.7.2) 
- seaborn (0.12.2)
``` 
We also provide detailed instructions for setting up the software environment with ease. Execute the following commands to install all required dependencies:

### Setting up the Software Environment
- Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ source ~/miniconda3/bin/activate
```
- Create the environment (This process takes about 3 minutes.)
```bash
$ source ./install.sh
```

# Running M-OFDFT

## Data and Model Preparation
We provide example scripts that run M-OFDFT for solving various molecular systems. To begin, download the necessary files from the [figshare](https://figshare.com/s/11e6445d581c06a58aea), which include the density functional model checkpoints and evaluation data. Then, extract the evaluation data into the `./data` directory and place the model checkpoints in the `./ckpts` directory. The resulting folder structure should look like:

```
|-- ckpts
    |-- Chignolin.pep5.MOFDFT.pt  
    |-- Ethanol.MOFDFT.pt  
    |-- QM9.MOFDFT.pt  
    |-- QMugs.bin1.MOFDFT.pt
|-- data
    |-- Chignolin
        |-- test_id.txt
        |-- input
    |-- Ethanol
        |-- test_id.txt
        |-- input
    |-- QM9
        |-- test_id.txt
        |-- input
    |-- QMugs
        |-- test_id.txt
        |-- input
|-- ofdft
    | ...
|-- scripts
    | ...
|-- README.md
|-- evaluate.py
|-- install.sh
|-- statistic.py
```

## Reproducing the Quantitative Results Reported in the Paper

Here we provide commands for reproducing quantitive results presented in the Results section of the paper. After executing these scripts, you can find the output files in `outputs/[Scripts name]`.

Note that the reference time values are provided for running M-OFDFT on a single structure, because in practice multiple computations can usually run simultaneously on powerful accelerators such as an Nvidia A100 GPU. To employ such parallelism, adjust the `NGPU` and `NWORKER` environment variables in the commands.

### Ethanol

The Ethanol data for evaluation can be found in `data/Ethanol`, which contains 10000 ethanol structures (and a reference ethanol structure) from the [MD17](http://www.sgdml.org/#datasets) dataset.

**This dataset can be used to reproduce the results of Fig. 2(a) (left panel).**

To reproduce the M-OFDFT results of Fig. 2(a), run M-OFDFT over the entire dataset with:
```bash
# adjust the `NGPU` and `NWORKER` environment variables to speed up the computation
$ NGPU=1 NWORKER=4 bash scripts/evaluate/examples/Ethanol.MOFDFT.sh
```

The reference time for running the residual KEDF variant of M-OFDFT on an ethanol structure is 29.7 seconds on an 80-GiB Nvidia A100 GPU. The expected output is:
```
------------------------------------------------------------
|Ethanol|path: outputs/Ethanol.MOFDFT/total.csv
|calculating metrics for 10001 molecules
|Ethanol|relative energy MAE: 0.22 kcal/mol
------------------------------------------------------------
```

To reproduce the results for classical KEDFs in Fig. 2(a), run the `Ethanol.classicKEDF.sh` scripts with a KEDF specification, including `TF`, `TFVW`, `TFVW1.1`(i.e., TF+1/9vW) and `APBE`:
```bash
$ NGPU=1 NWORKER=1 bash scripts/evaluate/examples/Ethanol.classicKEDF.sh [KEDF](TF|TFVW|TFVW1.1|APBE)
```

### QM9

The QM9 data for evaluation can be found in `data/QM9`, which contains 619 $\rm C_7H_{10}O_2$ isomer strucutures from the [QM9](http://dx.doi.org/10.6084/m9.figshare.978904) dataset.

**This dataset can be used to reproduce the results of Fig. 2(a) (right panel).**


To evaluate the entire dataset and reproduce the results of Fig. 2(a), run:
```bash
$ NGPU=1 NWORKER=1 bash scripts/evaluate/examples/QM9.MOFDFT.sh 
```

The reference time for running the residual KEDF variant of M-OFDFT on a QM9 structure is 51.2 seconds.
The expected results is:

```
------------------------------------------------------------
|QM9|path: outputs/QM9.MOFDFT/total.csv
|calculating metrics for 619 molecules
|QM9|relative energy MAE: 1.18 kcal/mol
------------------------------------------------------------
```  

To reproduce the results for classical KEDFs in Fig. 2(a), run the `QM9.classicKEDF.sh` scripts with a KEDF specification, including `TF`, `TFVW`, `TFVW1.1`(i.e., TF+1/9vW) and `APBE`:
```bash
$ NGPU=1 NWORKER=1 bash scripts/evaluate/examples/QM9.classicKEDF.sh [KEDF](TF|TFVW|TFVW1.1|APBE)
```

### QMugs

The QMugs data can be found in `data/QMugs`, which includes 850 test structures from the [QMugs](https://doi.org/10.3929/ethz-b-000482129) dataset, consisting of QMugs molecules with up to 100 heavy atoms, which are grouped according to the number of heavy atoms into bins of width 5. Therefore, each bin includes 50 structures.

**This dataset can be used to reproduce the results of Fig. 3(a).**

To evaluate over the entire dataset and reproduce the results of Fig. 3(a), run:
```bash
$ NGPU=1 NWORKER=1 bash scripts/evaluate/examples/QMugs.MOFDFT.sh
```

The expected output is (each line corresponds to one data point in Fig. 3(a).):
```
------------------------------------------------------------
|QMugs|path: ofdft/outputs/QMugs.MOFDFT/total.csv
|calculating metrics for 850 molecules
|QMugs|bin:2| per-atom absolute eng MAE: 0.06 kcal/mol
|QMugs|bin:3| per-atom absolute eng MAE: 0.07 kcal/mol
|QMugs|bin:4| per-atom absolute eng MAE: 0.10 kcal/mol
|QMugs|bin:5| per-atom absolute eng MAE: 0.12 kcal/mol
|QMugs|bin:6| per-atom absolute eng MAE: 0.17 kcal/mol
|QMugs|bin:7| per-atom absolute eng MAE: 0.14 kcal/mol
|QMugs|bin:8| per-atom absolute eng MAE: 0.19 kcal/mol
|QMugs|bin:9| per-atom absolute eng MAE: 0.11 kcal/mol
|QMugs|bin:10| per-atom absolute eng MAE: 0.11 kcal/mol
|QMugs|bin:11| per-atom absolute eng MAE: 0.13 kcal/mol
|QMugs|bin:12| per-atom absolute eng MAE: 0.10 kcal/mol
|QMugs|bin:13| per-atom absolute eng MAE: 0.09 kcal/mol
|QMugs|bin:14| per-atom absolute eng MAE: 0.10 kcal/mol
|QMugs|bin:15| per-atom absolute eng MAE: 0.07 kcal/mol
|QMugs|bin:16| per-atom absolute eng MAE: 0.07 kcal/mol
|QMugs|bin:17| per-atom absolute eng MAE: 0.11 kcal/mol
------------------------------------------------------------
``` 


### Chignolin

The Chignolin data can be found in `data/Chignolin`, which contains 50 test structures from https://www.deshawresearch.com/downloads/download_trajectory_science2011.cgi.

**This dataset can be used to reproduce the results of Fig. 3(c).**

To reproduce the results in Fig.3(c), run M-OFDFT over the entire dataset with: 
```bash
$ NGPU=1 NWORKER=1 bash scripts/evaluate/examples/Chignolin.MOFDFT.sh 
```

The reference time for running M-OFDFT on a Chignolin structure is 236.7 seconds.
The expected output is:

```
------------------------------------------------------------
|Chignolin|path: outputs/Chignolin.MOFDFT/total.csv
|calculating metrics for 50 molecules
|Chignolin|per-atom relative energy MAE: 0.07 kcal/mol
------------------------------------------------------------
```

To reproduce the results for classical KEDFs in Fig. 3(c), run the `Chignolin.classicKEDF.sh` scripts with a KEDF specification, including `TF`, `TFVW`, `TFVW1.1`(i.e., TF+1/9vW) and `APBE`:
```bash
$ NGPU=1 NWORKER=1 bash scripts/evaluate/examples/Chignolin.classicKEDF.sh [KEDF](TF|TFVW|TFVW1.1|APBE)
```


# Citation
If you use our code or method in your work, please consider citing the following:

```bibtex
@article{zhang2023m,
  title={M-OFDFT: Overcoming the Barrier of Orbital-Free Density Functional Theory for Molecular Systems Using Deep Learning},
  author={Zhang, He and Liu, Siyuan and You, Jiacheng and Liu, Chang and Zheng, Shuxin and Lu, Ziheng and Wang, Tong and Zheng, Nanning and Shao, Bin},
  journal={arXiv preprint arXiv:2309.16578},
  year={2023}
}
```