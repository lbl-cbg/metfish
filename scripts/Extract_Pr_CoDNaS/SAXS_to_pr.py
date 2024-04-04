# -*- coding: utf-8 -*-
"""
Author: Feng Yu. LBNL

Convert the SAXS file to P(r) curves using the RAW API

First version created on 01/04/2024
"""

import bioxtasraw.RAWAPI as raw
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(
    prog = 'SAXS2PR',
    description = 'Convert the SAXS file to P(r) curves using the RAW API' )
parser.add_argument('-f', '--filename', required=True)
parser.add_argument('-o', '--output', default='./')

args = parser.parse_args()

profiles_name=args.filename

# Load SAXS file (.dat format)
profiles = raw.load_profiles([profiles_name])

gi_profile=profiles[0]

# Convert SAXS file with Inverse Fourier Transform 
gi_bift = raw.bift(gi_profile)[0]

# Save the radius and P(r) to csv file
output_pd=pd.DataFrame({'r':gi_bift.r, 'P(r)':gi_bift.p},columns=['r','P(r)'])

output_loc=args.output
output_pd.to_csv(os.path.join(args.output,os.path.basename(profiles_name).split('.')[0]+".csv"),index=False)

