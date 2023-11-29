# retention_time_GNN
Pytorch implementation of the model described in the paper [Retention Time Prediction by Learning from Small Training Dataset with Pre-Trained Graph Neural Network](#)

## Components
- **data/*** - data files used
- **data/preprocessing.py** - script for data preprocessing
- **gnn/*.py** - GNN architecture
- **run_pretrain.py** - script for model pre-training
- **run_transfer.py** - script for model transfer learning and evaluation
- **dataset.py** - data structure & functions
- **model.py** - model training/inference functions
- **util.py**

## Data
- The METLIN-SMRT dataset can be downloaded from
  - https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913
- The target datasets from PredRet database can be downloaded from
  - http://predret.org/
- The target datasets from MoNA database can be downloaded from
  - https://mona.fiehnlab.ucdavis.edu/

## Usage Example
`python run_pretrain.py`
`python run_transfer.py -t FEM_long`

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Citation
```
@Article{Kwon2023,
  title={Retention Time Prediction through Learning from a Small Training Data Set with a Pretrained Graph Neural Network},
  author={Kwon, Youngchun and Kwon, Hyukju and Han, Jongmin and Kang, Myeonginn and Kim, Ji-Yeong and Shin, Dongyeeb and Choi, Youn-Suk and Kang, Seokho},
  journal={Analytical Chemistry},
  volume={95},
  number={47},
  pages={17273–17283},
  year={2023},
  doi={10.1021/acs.analchem.3c03177}
}
```
