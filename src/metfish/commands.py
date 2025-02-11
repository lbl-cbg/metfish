import argparse
import os
import sys

import pandas as pd

from .utils import get_Pr, extract_seq, sample_conformers, write_conformers
from pathlib import Path

def get_Pr_cli(argv=None):

    desc = "Calculate an exact P(r) curve for a given structure"
    epi = """By default, Dmax is calculated from the max distance found in the structure."""

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("struct_path", metavar="pdb|cif",
                        help="PDB or mmCIF file containing the model to use to calculate P(r)")
    parser.add_argument("id", help="the ID of the structure to use from the given file", nargs='?')
    parser.add_argument("-D", "--Dmax", type=float, default=None,
                        help="the max distance to consider when building P(r) histogram")
    parser.add_argument("-s", "--step", type=float, default=0.5,
                        help="the width of the bins to use when building P(r) histogram")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="the path to write the file to. by default, write to stdout")
    parser.add_argument("-f", "--force", action='store_true', help="overwrite output if it exist", default=False)

    args = parser.parse_args()

    out = sys.stdout
    if args.output is not None:
        if os.path.exists(args.output) and not args.force:
            print(f"File {args.output} exists. Exiting without overwriting. Use -f/--force to overwrite",
                  file=sys.stderr)
            return
        out = open(args.output, 'w')


    r, p = get_Pr(args.struct_path,
                  structure_id=args.id,
                  dmax=args.Dmax,
                  step=args.step)

    pd.DataFrame({"r": r, "P(r)": p}).to_csv(out, index=False)

def extract_seq_cli(argv=None):
    
    parser = argparse.ArgumentParser(
    description = '''Extract Sequence from PDB file.
    The output will be stored at the current directory as a fasta file.''' )
    
    parser.add_argument('-f', '--filename', required=True, help="input pdb file")
    parser.add_argument('-o', '--output', required=True, help="output fasta file path + name")

    args = parser.parse_args()

    extract_seq(args.filename,args.output)

def generate_nma_conformers_cli(argv=None):
    parser = argparse.ArgumentParser(
    description = '''Generate synthetic structures using normal mode analysis and traversing 
    the calculated nodes using a maximum RMSD.''' )
    
    parser.add_argument('-f', '--filename', required=True, help="input pdb file")
    parser.add_argument('-o', '--output', required=True, help="output directory for generated files")
    parser.add_argument('-n', '--n_modes', default=2, help="number of modes to use for NMA")
    parser.add_argument('-c', '--n_confs', default=6, help="number of conformers to generate")
    parser.add_argument('-r', '--rmsd', default=3, help="maximum RMSD to traverse nodes")

    args = parser.parse_args()

    fname = Path(args.filename)
    name = '_'.join(fname.stem.split('_')[1:])  # TODO - check this is accurate

    conformer_coords = sample_conformers(fname, 
                                         n_modes=args.n_modes, 
                                         n_confs=args.n_confs, 
                                         rmsd=args.rmsd)

    write_conformers(Path(args.output), name, conformer_coords)
