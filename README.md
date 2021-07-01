# _DeepAb_
This is the official repository containing the models and code for the [DeepAb antibody structure prediction method](https://www.biorxiv.org/content/10.1101/2021.05.27.445982v1.full).

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
wget https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz
tar -xf ensemble_abresnet_v1.tar.gz -C trained_models/
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

### Design scoring
Calculate ΔCCE for list of designed sequences:
```
python score_design.py data/sample_files/wt.fasta data/sample_files/h_mut_seqs.fasta data/sample_files/l_mut_seqs.fasta design_out.csv
```

## References
[1] JA Ruffolo, J Sulam, and JJ Gray. "Antibody structure prediction using interpretable deep learning." _bioRxiv_ (2021).
