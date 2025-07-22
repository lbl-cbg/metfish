from Bio.PDB import PDBParser, PDBIO
import os
import requests
import argparse

def separate_nmr_frames(input_pdb, chain_id='A', pdb_id=None, output_dir='separated_frames'):
    """
    Separate NMR PDB file with multiple frames into individual PDB files for a specific chain.
    
    Args:
        input_pdb (str): Path to input PDB file
        chain_id (str): Chain ID to extract
        output_dir (str): Directory to save separated frames
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("nmr_structure", input_pdb)
    
    # Initialize PDB writer
    io = PDBIO()
    
    # Iterate through models (frames)
    for model_id, model in enumerate(structure):
        # Check if the specified chain exists in this model
        if chain_id in model:
            # Set structure to current model's specific chain
            io.set_structure(model[chain_id])
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{pdb_id}_{chain_id}_{model_id + 1}.pdb")
            
            # Save the chain
            io.save(output_file)
            print(f"Saved chain {chain_id} frame {model_id + 1} to {output_file}")
        else:
            print(f"Chain {chain_id} not found in frame {model_id + 1}")

# Example usage
if __name__ == "__main__":
    # Replace with your NMR PDB file path
    def main():
        parser = argparse.ArgumentParser(description='Separate NMR PDB file frames into individual files')
        parser.add_argument('-i', '--input', dest='pdb_names', nargs='+', required=True, help='PDB names (e.g., 1AEL-12_A 2ABC-5_B)')
        parser.add_argument('-o', '--output', dest='output_dir', default='separated_frames', help='Output directory for separated frames')
        args = parser.parse_args()
        
        for pdb_input in args.pdb_names:
            pdb_id = pdb_input[:4].lower()  # Extract first 4 characters as PDB ID
            chain_id = pdb_input.split('_')[1] if '_' in pdb_input else 'A'  # Extract chain ID
            
            # Download PDB file
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)
            
            if response.status_code == 200:
                input_file = f"{pdb_id}.pdb"
                with open(input_file, 'w') as f:
                    f.write(response.text)
                print(f"Downloaded {input_file}")
                
                separate_nmr_frames(input_file, chain_id=chain_id, pdb_id=pdb_id, output_dir=args.output_dir)
            else:
                print(f"Failed to download PDB {pdb_id}")

    if __name__ == "__main__":
        main()
