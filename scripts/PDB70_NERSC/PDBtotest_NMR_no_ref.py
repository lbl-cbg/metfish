# %%
import Bio.Align as Align
import Bio.SeqIO as SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBList
import argparse
import os

# %%
def extract_seq(input_pdb_path, chain_ID):
    counter = 0
    # Parse all chains and filter by chain_ID
    for record in SeqIO.parse(input_pdb_path, "pdb-atom"):
        if record.id.split(':')[1] == chain_ID:
            if counter > 0:
                raise ValueError("More than 1 matching chain {} found in file {}".format(chain_ID, input_pdb_path))
            sequence_to_be_fixed = record.seq
            counter += 1

    if counter == 0:
        raise ValueError("Chain {} not found in file {}".format(chain_ID, input_pdb_path))

    return sequence_to_be_fixed
    

# %%
# function to save the sequence to fast
def save_fasta(sequence, input_name ,output_path):

    new_seq_record = SeqRecord(sequence, id=input_name, description='')
    SeqIO.write(new_seq_record, os.path.join(output_path, input_name +'.fasta') ,"fasta")
# Get pdb name to save files
def extract_pdb_name(input_pdb_path):

    base_name_without_extension = os.path.splitext(os.path.basename(input_pdb_path))[0]
    ref_pdb_name=base_name_without_extension.split('_')[0].lower()
    chain_ID=base_name_without_extension.split('_')[1]

    print("Base name without extension: ", base_name_without_extension)
    print("Reference PDB name: ", ref_pdb_name)
    print("Chain ID: ", chain_ID)
    
    return base_name_without_extension,ref_pdb_name, chain_ID 
# the "entire workflow"
def extract_fixed_seq(input_pdb_path,output_path):
    base_name_without_extension, ref_pdb_name, chain_ID = extract_pdb_name(input_pdb_path)
    sequence_to_be_fixed = extract_seq(input_pdb_path, chain_ID)
    save_fasta(sequence_to_be_fixed, base_name_without_extension ,output_path)
# Command line interface
def main():
    parser = argparse.ArgumentParser(
    prog = 'NMR PDB Sequence Extractor',
    description = '''Test whether the NMR PDB has missing loops. If not output the fasta.''' )
    
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-o', '--output', default='./')

    args = parser.parse_args()
    extract_fixed_seq(args.filename, args.output)
    #print("The full sequence of {} is {}".format(args.filename,full_seq))

if __name__ == '__main__':
    main()
