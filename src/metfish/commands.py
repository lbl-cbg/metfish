import argparse
import os
import sys

import pandas as pd

from .utils import get_Pr
from .preprocess import prep_conformer_pairs

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

def prep_conformer_pairs_cli(argv=None):

    desc = "Preprocess conformer pair pdb files to generate fasta sequences, atom positions in AF structure format"
    epi = """Requires folder with the csv of apo/holo pairs and pdbs from Saldano et al., 2022"""

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("data_dir", help="the directory containing the pdb files / csv")
    parser.add_argument("-o", "--output_dir", type=str, default=None, 
                        help="the directory to save output files to. Defaults to data_dir if not provided")
    parser.add_argument("-n", "--n_pairs", type=int, default=6, 
                        help="the # of pairs to look at, N pairs -> 2N structures predicted")
    parser.add_argument("-a", "--af_output_dir", type=str, default=None,
        help="path with alphafold outputs, used to calculate which conformer is less similar to the original AF prediction.",
    )

    args = parser.parse_args()

    prep_conformer_pairs(args.data_dir,
                         output_dir=args.output_dir,
                         n_pairs=args.n_pairs,
                         af_output_dir=args.af_output_dir,)