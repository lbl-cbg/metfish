# AlphaSAXS

**Experimental Data Driven AI Framework for Flexible Protein Conformational Reconstruction**

Feng Yu, Stephanie Prince, Andrew Tritt, Kanupriya Pande, Greg L. Hura, Oliver Ruebel, Susan E. Tsutakawa  
*Molecular Biophysics and Integrated Bioimaging / Computational Biosciences Group, Lawrence Berkeley National Laboratory*  
Correspondence: setsutakawa@lbl.gov, oruebel@lbl.gov

---

Deep learning has transformed structural biology, enabling the prediction of static protein folds from primary sequences with near-experimental accuracy. In reality, most proteins are intrinsically dynamic, shifting between conformations to carry out their diverse functions, and current sequence-only models often fail to capture the specific conformational states and heterogeneity dictated by cellular environments or ligand binding. While recent generative models can sample broad conformational landscapes, they remain unconstrained by physical reality, often hallucinating plausible but experimentally invalid states.

Here, we present **AlphaSAXS**, an end-to-end framework that constrains AI inference using Small Angle X-ray Scattering (SAXS) experimental solution scattering data. By integrating real-space pair distance distributions P(r) directly into both the multiple sequence alignment (MSA) and pair representation modules in the AlphaFold/OpenFold architecture, AlphaSAXS effectively steers structural hypotheses toward experimentally observed structures. We demonstrate that AlphaSAXS improves prediction accuracy where sequence-only models fail to capture diverse conformations in apo-holo transitions, successfully guiding predictions with experimental scattering profiles.

This work establishes a paradigm for experimentally guided AI, bridging the gap between probabilistic sampling and biophysical measurement, with implications for understanding allosteric mechanisms and development of novel therapeutics, bioengineering and biomanufacturing, and biodefense.

---

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
python src/metfish/msa_model/train.py path/to/data path/to/output
```

Note that the training pipeline expects the following directory structure:

```
data_dir/
├── pdb/
│   └── {pdb_prefix}{name}{pdb_ext}  # e.g., fixed_1ABC_A.pdb
├── saxs_r/
│   └── {name}{saxs_ext}   # SAXS P(r) data files
├── msa/
│   └── a3m
│     └── {msa_id}.a3m     # MSA data files
└── scripts/
    ├── input_training.csv
    └── input_validation.csv
```

- **CSV files** (`input_training.csv`, `input_validation.csv`): Must contain:
  - `name` - Protein identifier (used to locate corresponding files)
  - `seqres` - Protein amino acid sequence
  - `msa_id` (optional) - MSA identifier if different from name

- **PDB files**: Standard PDB format structure files
  - Default naming: `fixed_{name}.pdb` (prefix can vary based on dataset)
  - The script auto-detects if using simulated data and adjusts paths accordingly

- **SAXS files**: CSV format containing P(r) distribution data
  - Must include a `P(r)` column with the pair distance distribution
  - Common naming patterns: `{name}.pdb.pr.csv` or `{name}.pr.csv`
  - The script auto-detects the extension pattern from existing files

- **MSA directories**: Each protein should have a subdirectory in `msa/` containing:
  - **A3M files** (`.a3m`): Multiple sequence alignments in A3M format structured within an `a3m` subfolder.

## Reproducing Publication Figures

All publication figures can be regenerated from pre-computed results using the master figure generation script. No model inference is required.

```bash
python src/metfish/analysis/scripts/generate_all_figures.py --output-dir publication_figures/
```

This regenerates all figures from Figures 2–4 and the supplementary information into the specified output directory. To generate a subset:

```bash
# Figure 2 only (model comparison: RMSD, Rg, SAXS, metrics)
python src/metfish/analysis/scripts/generate_all_figures.py --figures figure2

# Figure 3 only (apo-holo analysis: P(r), RMSD, recovery, correlation)
python src/metfish/analysis/scripts/generate_all_figures.py --figures figure3

# Figure 4 only (ensemble analysis: barplot, Rg, Re, diversity)
python src/metfish/analysis/scripts/generate_all_figures.py --figures figure4
```

Supplementary Information figures (all 80 proteins) can be generated with:

```bash
python src/metfish/analysis/scripts/generate_si_figures.py --output-dir SI_figures/
```

## Data Availability

Data will be available upon request.

## Generating Figures

The `scripts/generate_figures.py` script runs model inference and generates comparison visualizations for protein structure predictions. This script processes multiple models (AlphaFold, SFold_NMR, SFold_NMA) and creates figures comparing their performance.

### Basic Usage

```bash
python scripts/generate_figures.py \
  --data-dir /path/to/data \
  --ckpt-dir /path/to/checkpoints \
  --output-dir /path/to/output \
  --skip-inference false
```

Note that the inference pipeline expects the following directory structure:

```
data_dir/
├── input_all.csv          # CSV with protein names and sequences
├── pdbs/
│   └── {name}.pdb         # PDB structure files
├── saxs_r/
│   └── {name}{saxs_ext}   # SAXS P(r) data files
├── msa/
│   └── a3m
│     └── {msa_id}.a3m     # MSA data files
```

**File Format Requirements:**

- **CSV file** (`input_all.csv`): Must contain the following columns:
  - `name` - Protein identifier (used to locate corresponding files)
  - `seqres` - Protein amino acid sequence
  - `msa_id` (optional) - MSA identifier if different from name

- **PDB files**: Standard PDB format structure files
  - Default naming: `{name}.pdb`
  - Can be customized with `--pdb-ext` flag

- **SAXS files**: CSV format containing P(r) distribution data
  - Must include a `P(r)` column with the pair distance distribution
  - Default naming: `{name}_atom_only.csv` or `{name}.pdb.pr.csv`
  - Can be customized with `--saxs-ext` flag

- **MSA directories**: Each protein should have a subdirectory in `msa/` containing:
  - **A3M files** (`.a3m`): Multiple sequence alignments in A3M format structured within an `a3m` subfolder.


## Commands

The following CLI commands are available after installation:

- `calc-pr`: Calculate the P(r) pair distance distribution for a protein structure from a PDB or mmCIF file.
  ```bash
  calc-pr structure.pdb
  ```
- `extract-seq`: Extract the amino acid sequence from a PDB file.
  ```bash
  extract-seq structure.pdb
  ```
- `generate-nma-conformers`: Generate an ensemble of conformations by normal mode analysis (NMA).
  ```bash
  generate-nma-conformers structure.pdb --n-modes 10 --n-conformers 50
  ```
