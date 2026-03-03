# Context-aware probabilistic ship trajectory forecasting

## Requirements
To install all the requirements, one needs to first install:
+ conda

#### linux server
The proper installation must then be done with conda.

conda create -n nereus_env python=3.11 -y
conda activate nereus_env

export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"
export PIP_FIND_LINKS="https://data.pyg.org/whl/torch-2.5.1+cu124.html"
pip install -e .

## Structure

data: DataLoader (AIS trajectories) and SceneLoader (Context Maps, Density Maps, ...)
eval: full evaluation scripts and eval functions used for tuning
models:
  - desire: implementation adapted from https://github.com/AkashGanesan/desire-pytorch
  - gmm: clutering to generate density maps
  - gru: baseline model
  - nereus: main model with its map-, interaction-, and prior-modules
  - traisfromer: implementation adapted from https://github.com/CIA-Oceanix/TrAISformer
plots: loose plot funktions 
train: full training and tuning scripts
utils: config and logger

## Data

data from preprocessing 
see https://cau-git.rz.uni-kiel.de/inf/intern/ag-tomforde/gfalouji/autonomous.maritime/nereus/ais.processing