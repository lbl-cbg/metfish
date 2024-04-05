from pathlib import Path
from metfish.utils import extract_seq


def __run_test(pdb_name):
    # extract the sequence from the pdb file
    extract_seq(f'tests/data/{pdb_name}.pdb', f'tests/data/{pdb_name}_temp.fasta')
    
    with open(f'tests/data/{pdb_name}_temp.fasta', 'r') as file:
        seq_created = file.readlines()
    with open(f'tests/data/{pdb_name}.fasta', 'r') as file:
        seq_reference = file.readlines()

    # check reference sequence and created sequence are same
    assert seq_reference[1] == seq_created[1]

    # remove the temporary file
    Path.unlink(f"tests/data/{pdb_name}_temp.fasta")

def test_seq():
    pdb_name = "3nir"
    __run_test(pdb_name)

def test_seq_with_residue_gap():
    pdb_name = "3DB7_A"
    __run_test(pdb_name)

def test_seq_with_out_of_order_residues():
    pdb_name = "3M8J_A"
    __run_test(pdb_name)

def test_seq_with_multiple_chains():
    pdb_name = "5K1S_1"
    try:
        extract_seq(f'tests/data/{pdb_name}.pdb', "tests/data")
    except ValueError as e:
        assert str(e) == f"More than 1 Chain is in the file tests/data/{pdb_name}.pdb"
    else:
        assert False