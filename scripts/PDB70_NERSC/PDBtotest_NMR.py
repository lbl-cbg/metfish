# %%
import Bio.Align as Align
import Bio.SeqIO as SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBList
import argparse
import os

# %%
def extract_ref_seq(input_pdb_path, ref_pdb_name, chain_ID):
    pdbl = PDBList()
    seqres_type="pdb-seqres"
    # Retrieve PDB reference (with full sequence) from RCSB
    # Will not rewrite PDB if already downloaded
    # Will download obsolete PDBS
    ref_file_name=pdbl.retrieve_pdb_file(ref_pdb_name.upper(), file_format='pdb', pdir="./tmp",obsolete=True)
    if not os.path.exists(ref_file_name):
        ref_file_name=pdbl.retrieve_pdb_file(ref_pdb_name, file_format='mmCif', pdir="./tmp")
        seqres_type="cif-seqres"
    if not os.path.exists(ref_file_name):
        raise ValueError("PDB doesn't exist in the database")
    for record in SeqIO.parse(ref_file_name, seqres_type):
        # Get the corresponding chain from the Reference PDB and extract sequence
        if record.annotations["chain"] == chain_ID:
            break
    ref_seq=record.seq
    print(ref_seq,seqres_type)
    #os.remove(ref_file_name)

    counter=1
    # To prevent 2 chains in the "to be fixed" PDB file
    for record2 in SeqIO.parse(input_pdb_path,"pdb-atom"):
        if counter > 1:
            raise ValueError("More than 1 Chain is in the file {}".format(input_pdb_path))
        else:
            sequence_to_be_fix=record2.seq
        counter+=1
    return sequence_to_be_fix, ref_seq
    

# %%
# function to save the sequence to fast
def save_fasta(sequence, input_name ,output_path):

    new_seq_record = SeqRecord(sequence, id=input_name, description='')
    SeqIO.write(new_seq_record, os.path.join(output_path,+ input_name + '.fasta') ,"fasta")
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
    sequence_to_be_fix, ref_seq = extract_ref_seq(input_pdb_path, ref_pdb_name, chain_ID)
    if sequence_to_be_fix == ref_seq:
        print("The sequence matches the SEQRES")
        save_fasta(sequence_to_be_fix, base_name_without_extension ,output_path)
    else:
        print("The sequence does not match the SEQRES")
        print("The sequence to be fixed is {}".format(sequence_to_be_fix))
        print("The reference sequence is {}".format(ref_seq))
        raise ValueError("The sequence does not match the SEQRES. It is not suitable for test dataset.")
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
