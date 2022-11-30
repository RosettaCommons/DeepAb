# _DeepAb_
Official repository for [DeepAb](https://www.sciencedirect.com/science/article/pii/S2666389921002804): Antibody structure prediction using interpretable deep learning.  The code, data, and weights for this work are made available under the [Rosetta-DL license](LICENSE.md) as part of the [Rosetta-DL](https://github.com/RosettaCommons/Rosetta-DL) bundle.

Try antibody structure prediction in [Google Colab](https://colab.research.google.com/github/RosettaCommons/DeepAb/blob/main/DeepAb.ipynb).

## Setup

_Optional_: Create and activate a python virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
Install project dependencies
```
pip install -r requirements.txt
```

_Note_: PyRosetta should be installed following the instructions [here](http://pyrosetta.org/downloads).

Download pretrained model weights
```
wget https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz
tar -xf ensemble_abresnet_v1.tar.gz
```
After unzipping, pre-trained models might need to be moved such that they have paths `trained_models/ensemble_abresnet/rs*.pt`

## Common workflows

Additional options for all scripts are available by running with `--help`.

_Note_: This project is tested with Python 3.7.9

_Note_: Using `--renumber` option will send your antibody to the [AbNum server](http://www.bioinf.org.uk/abs/abnum/). If working with confidential sequences you should avoid this option and use an external renumbering tool.


### Structure prediction
Generate an antibody structure prediction from an Fv sequence with five decoys:
```
python predict.py data/sample_files/4h0h.fasta --decoys 5 --renumber
```
Generate a structure for a single heavy or light chain:
```
python predict.py data/sample_files/4h0h.fasta --decoys 5 --single_chain
```
_Note_: The fasta file should contain a single entry labeled "H" (even if the sequence is a light chain).

**Expected output**

After the script completes, the final prediction will be saved as `pred.deepab.pdb`.  The numbered decoy structures will be stored in the `decoys/` directory.


### Attention annotation
Annotate an Fv structure with H3 attention:
```
python annotate_attention.py data/sample_files/4h0h.truncated.pdb --renumber --cdr_loop h3
```
_Note_: CDR loop residues are determined using Chothia definitions, so the input structure should be numbered beforehand or renumbered by passing `--renumber`

**Expected output**

After the script completes, the annotated PDB will overwrite the input file (unless `--out_file` is specificed).  Annotations will be stored as b-factor information, and can be visualized in [PyMOL](https://pymol.org/2/) or similar software.

### Design scoring
Calculate ΔCCE for list of designed sequences:
```
python score_design.py data/sample_files/wt.fasta data/sample_files/h_mut_seqs.fasta data/sample_files/l_mut_seqs.fasta design_out.csv
```

**Expected output**

After the script completes, the designs and scores will be written to a CSV file with each row containing the design ID, heavy chain sequence, light chain sequence, and  ΔCCE value.

## References
[1] JA Ruffolo, J Sulam, and JJ Gray. "[Antibody structure prediction using interpretable deep learning.](https://www.sciencedirect.com/science/article/pii/S2666389921002804)" _Patterns_ (2022).
