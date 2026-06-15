"""
Tests for CLI commands in metfish.commands
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, mock_open
import pandas as pd

from metfish.commands import get_Pr_cli, extract_seq_cli, generate_nma_conformers_cli


class TestGetPrCli:
    """Tests for get_Pr_cli command."""
    
    @pytest.mark.requires_data
    def test_get_pr_cli_basic(self, test_data_dir, temp_dir, capsys):
        """Test basic P(r) calculation via CLI."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        test_args = ['get_Pr', str(pdb_path)]
        with patch.object(sys, 'argv', test_args):
            get_Pr_cli()
        
        captured = capsys.readouterr()
        assert 'r,' in captured.out or 'P(r)' in captured.out
    
    @pytest.mark.requires_data
    def test_get_pr_cli_with_output(self, test_data_dir, temp_dir):
        """Test P(r) calculation with output file."""
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "pr_output.csv"
        
        test_args = ['get_Pr', str(pdb_path), '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            get_Pr_cli()
        
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert 'r' in df.columns
        assert 'P(r)' in df.columns
    
    @pytest.mark.requires_data
    def test_get_pr_cli_with_dmax(self, test_data_dir, temp_dir):
        """Test P(r) calculation with custom Dmax."""
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "pr_output.csv"
        
        test_args = ['get_Pr', str(pdb_path), '-D', '50', '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            get_Pr_cli()
        
        df = pd.read_csv(output_path)
        assert df['r'].max() <= 50
    
    @pytest.mark.requires_data
    def test_get_pr_cli_with_step(self, test_data_dir, temp_dir):
        """Test P(r) calculation with custom step size."""
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "pr_output.csv"
        
        test_args = ['get_Pr', str(pdb_path), '-s', '1.0', '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            get_Pr_cli()
        
        df = pd.read_csv(output_path)
        # Check that step size is approximately 1.0
        steps = df['r'].diff().dropna()
        assert all(abs(s - 1.0) < 0.01 for s in steps)
    
    @pytest.mark.requires_data
    def test_get_pr_cli_force_overwrite(self, test_data_dir, temp_dir):
        """Test force overwrite of existing file."""
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "pr_output.csv"
        
        # Create existing file
        output_path.write_text("existing content")
        
        # Without force, should not overwrite
        test_args = ['get_Pr', str(pdb_path), '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            get_Pr_cli()
        
        # With force, should overwrite
        test_args = ['get_Pr', str(pdb_path), '-o', str(output_path), '-f']
        with patch.object(sys, 'argv', test_args):
            get_Pr_cli()
        
        # Check that file was overwritten
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert 'r' in df.columns


class TestExtractSeqCli:
    """Tests for extract_seq_cli command."""
    
    @pytest.mark.requires_data
    def test_extract_seq_cli_basic(self, test_data_dir, temp_dir):
        """Test basic sequence extraction via CLI."""
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "output.fasta"
        
        test_args = ['extract_seq', '-f', str(pdb_path), '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            extract_seq_cli()
        
        assert output_path.exists()
        with open(output_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) >= 2
        assert lines[0].startswith('>')
    
    @pytest.mark.requires_data
    def test_extract_seq_cli_with_gaps(self, test_data_dir, temp_dir):
        """Test sequence extraction with gaps via CLI."""
        pdb_path = test_data_dir / "3DB7_A.pdb"
        output_path = temp_dir / "output.fasta"
        
        test_args = ['extract_seq', '-f', str(pdb_path), '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            extract_seq_cli()
        
        with open(output_path, 'r') as f:
            created_seq = f.readlines()[1].strip()
        
        with open(test_data_dir / "3DB7_A.fasta", 'r') as f:
            reference_seq = f.readlines()[1].strip()
        
        assert created_seq == reference_seq
    
    @pytest.mark.requires_data
    def test_extract_seq_cli_multiple_chains(self, test_data_dir, temp_dir):
        """Test that multiple chains raise error via CLI."""
        pdb_path = test_data_dir / "5K1S_1.pdb"
        output_path = temp_dir / "output.fasta"
        
        test_args = ['extract_seq', '-f', str(pdb_path), '-o', str(output_path)]
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(ValueError, match="More than 1 Chain"):
                extract_seq_cli()


class TestGenerateNmaConformersCli:
    """Tests for generate_nma_conformers_cli command."""
    
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_generate_nma_conformers_basic(self, test_data_dir, temp_dir):
        """Test basic NMA conformer generation via CLI."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        test_args = [
            'generate_nma',
            '-f', str(pdb_path),
            '-o', str(temp_dir),
            '-n', '1',
            '-c', '2',
            '-r', '1.0'
        ]
        with patch.object(sys, 'argv', test_args):
            generate_nma_conformers_cli()
        
        # Check that output files were created
        output_files = list(temp_dir.glob('*.pdb'))
        assert len(output_files) > 0
    
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_generate_nma_conformers_multiple_modes(self, test_data_dir, temp_dir):
        """Test NMA conformer generation with multiple modes."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        test_args = [
            'generate_nma',
            '-f', str(pdb_path),
            '-o', str(temp_dir),
            '-n', '2',
            '-c', '4',
            '-r', '2.0'
        ]
        with patch.object(sys, 'argv', test_args):
            generate_nma_conformers_cli()
        
        output_files = list(temp_dir.glob('*.pdb'))
        # Should have multiple conformers
        assert len(output_files) > 1


class TestCommandEdgeCases:
    """Tests for edge cases and error handling in commands."""
    
    def test_get_pr_cli_invalid_file(self, temp_dir):
        """Test get_Pr_cli with non-existent file."""
        test_args = ['get_Pr', 'nonexistent.pdb']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(Exception):
                get_Pr_cli()
    
    def test_extract_seq_cli_missing_args(self):
        """Test extract_seq_cli with missing arguments."""
        test_args = ['extract_seq']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                extract_seq_cli()
    
    def test_generate_nma_cli_missing_args(self):
        """Test generate_nma_conformers_cli with missing arguments."""
        test_args = ['generate_nma']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                generate_nma_conformers_cli()