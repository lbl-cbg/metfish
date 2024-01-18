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

## Commands

- `calc-pr`: Calculate P(r) for models from a PDB or mmCIF file
