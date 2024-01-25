from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import argparse
import os

def extract_seq(pdb_input,output_path):
    pdb_name=os.path.basename(pdb_input).split(".")[0]
    counter=1
    for record in SeqIO.parse(pdb_input,"pdb-atom"):
        if counter > 1:
            raise ValueError("More than 1 Chain is in the file {}".format(pdb_input))
        else:
            new_seq_record = SeqRecord(record.seq, id=pdb_name, description='')
            SeqIO.write(new_seq_record, output_path ,"fasta")
        counter+=1

def main():
    parser = argparse.ArgumentParser(
    prog = 'PDBtoSeq',
    description = '''Extract Sequence from PDB using the BioPython API.
    The output will be stored at the current directory as a fasta file.''' )
    
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-o', '--output', default='./')

    args = parser.parse_args()

    extract_seq(args.filename,args.output)

if __name__ == '__main__':
    main()
