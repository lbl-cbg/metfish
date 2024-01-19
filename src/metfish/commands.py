import argparse
import os
import sys

import pandas as pd

from .utils import get_Pr

def get_Pr_cli(argv=None):

    desc = "Calculate an exact P(r) curve for a given structure"
    epi = """By default, Dmax is calculated from the max distance foudn in the structure."""

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("struct_path", metavar="pdb|cif", help="PDB or mmCIF file containing the model to use to calculate P(r)")
    parser.add_argument("id", help="the ID of the structure to use from the given file", nargs='?')
    parser.add_argument("-D", "--Dmax", type=float, help="the max distance to consider when building P(r) histogram", default=None)
    parser.add_argument("-s", "--step", type=float, help="the width of the bins to use when building P(r) histogram", default=0.5)
    parser.add_argument("-o", "--output", type=str, help="the path to write the file to. by default, write to stdout", default=None)
    parser.add_argument("-f", "--force", action='store_true', help="overwrite output if it extist", default=False)

    args = parser.parse_args()

    out = sys.stdout
    if args.output is not None:
        if os.path.exists(args.output) and not args.force:
            print(f"File {args.output} exists. Exiting without overwritting. Use -f/--force to overwrite", file=sys.stderr)
            return
        out = open(args.output, 'w')


    r, p = get_Pr(args.struct_path,
                  structure_id=args.id,
                  dmax=args.Dmax,
                  step=args.step)

    pd.DataFrame({"r": r, "P(r)": p}).to_csv(out, index=False)
