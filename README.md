# metfish
Code for the MetFish LDRD project

## Installation

The following commands will install the code in this repository in such a way
that will allow one to use the tools provided by said code. With that said, 
the provided sequence of commands may not suit your specific needs.
As this repository follows PEP 517 style packaging, there are many 
ways to install the software, so please use discretion and adapt as necessary.

```bash
git clone git@github.com:lbl-cbg/metfish.git
cd metfish
pip install -r requirements.txt
pip install .
```

### Training the model
Training the modified OpenFold model requires some additional dependencies and constraints. Please follow these instructions to setup a conda environment for model training and evaluation. Note the OpenFold installation requires CUDA 11.

```bash
conda create -n metfish python=3.9
conda activate metfish
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html  # install separately to specify with findlinks
pip install .[training,viz]
```

To install additional resources required by OpenFold, run this command and copy to the openfold/resources folder:
```bash
wget -N --no-check-certificate -P openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

To run training, use the following command, which provides several additional flags:
```bash
python train.py path/to/data path/to/output
```

## Commands

- `calc-pr`: Calculate P(r) for models from a PDB or mmCIF file
- `extract-seq`: Extract Sequence from a PDB file.
- `generate-nma-conformers`: Generate conformations from normal mode analysis