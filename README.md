# _DeepAb_: Work in progress
This repository contains the models and code used for the DeepAb antibody structure prediction method.

## Setup

_Note_: This project is tested with Python 3.7.9

_Optional_: Create and activate a python virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
Install project dependencies
```
pip install -r requirements
```
Download pretrained model weights
```
wget link.to/come
```

## Common workflows

Additional options for all scripts are available by running with `--help`.

### Structure prediction
Generate an antibody structure prediction from an Fv sequence with five decoys:
```
python predict.py data/sample_files/4h0h.fasta --decoys 5 --renumber
```
Generate Rosetta constraint files for an Fv sequence:
```
python predict.py data/sample_files/4h0h.fasta --decoys 0 --keep_constraints
```

### Attention annotation
Annotate an Fv structure with H3 attention:
```
python annotate_attention.py data/sample_files/4h0h.truncated.pdb --renumber --cdr_loop h3
```
_Note_: CDR loop residues are determined using Chothia definitions, so the input structure should be numbered beforehand or renumbered by passing `--renumber`

## References
[1] JA Ruffolo, J Sulam, and JJ Gray. "Antibody structure prediction using interpretable deep learning." _bioRxiv_ (2021).