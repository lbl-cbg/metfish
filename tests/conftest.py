"""
Pytest configuration and fixtures for the metfish test suite.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_pdb_path(test_data_dir):
    """Return path to a sample PDB file."""
    return test_data_dir / "3nir.pdb"


@pytest.fixture(scope="session")
def sample_cif_path(test_data_dir):
    """Return path to a sample CIF file."""
    return test_data_dir / "3nir.cif"


@pytest.fixture(scope="session")
def sample_fasta_path(test_data_dir):
    """Return path to a sample FASTA file."""
    return test_data_dir / "3nir.fasta"


@pytest.fixture
def sample_structure(sample_pdb_path):
    """Load and return a sample BioPython structure."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure("test", str(sample_pdb_path))


@pytest.fixture
def sample_pdb_df(sample_pdb_path):
    """Load and return a sample PDB as a pandas DataFrame."""
    return PandasPdb().read_pdb(str(sample_pdb_path)).df['ATOM']


@pytest.fixture
def sample_coordinates():
    """Return sample protein coordinates for testing."""
    # Simple 3-residue test structure
    return np.array([
        [0.0, 0.0, 0.0],   # CA
        [1.5, 0.0, 0.0],   # N
        [0.0, 1.5, 0.0],   # C
        [3.8, 0.0, 0.0],   # CA
        [5.3, 0.0, 0.0],   # N
        [3.8, 1.5, 0.0],   # C
        [7.6, 0.0, 0.0],   # CA
        [9.1, 0.0, 0.0],   # N
        [7.6, 1.5, 0.0],   # C
    ])


@pytest.fixture
def sample_weights():
    """Return sample atomic weights for testing."""
    # Typical weights for CA, N, C atoms
    return np.array([6, 7, 6, 6, 7, 6, 6, 7, 6])


@pytest.fixture
def temp_dir(tmp_path):
    """Return a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mock_saxs_profile():
    """Return a mock SAXS profile for testing."""
    r = np.arange(0, 128.5, 0.5)
    p = np.exp(-0.1 * r) * np.sin(r / 10)
    p = p / p.sum()  # Normalize
    return r, p


@pytest.fixture(scope="session")
def pytorch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def openfold_available():
    """Check if OpenFold is available."""
    try:
        import openfold
        return True
    except ImportError:
        return False


@pytest.fixture
def sample_tensor():
    """Return a sample tensor for testing."""
    if not torch.cuda.is_available():
        return torch.randn(2, 10, 3)
    return torch.randn(2, 10, 3).cuda()


@pytest.fixture
def sample_batch_features():
    """Return sample batch features for model testing."""
    batch_size = 2
    seq_len = 10
    
    features = {
        'aatype': torch.randint(0, 20, (batch_size, seq_len)),
        'residue_index': torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
        'seq_length': torch.tensor([seq_len] * batch_size),
        'all_atom_positions': torch.randn(batch_size, seq_len, 37, 3),
        'all_atom_mask': torch.ones(batch_size, seq_len, 37),
    }
    
    return features


@pytest.fixture
def skip_if_no_torch(pytorch_available):
    """Skip test if PyTorch is not available."""
    if not pytorch_available:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_if_no_openfold(openfold_available):
    """Skip test if OpenFold is not available."""
    if not openfold_available:
        pytest.skip("OpenFold not available")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    except:
        pass


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown markers."""
    config.addinivalue_line("markers", "requires_data: mark test as requiring local test data files")
    config.addinivalue_line("markers", "requires_torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "requires_openfold: mark test as requiring OpenFold")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on availability of optional heavy dependencies or test data.

    This inspects collected tests and applies skip markers for tests that declare
    `requires_torch`, `requires_openfold`, or `requires_data` when the corresponding
    resource is not available. This prevents noisy failures in environments that
    intentionally don't have heavy ML deps installed.
    """
    from pathlib import Path
    import importlib

    has_torch = importlib.util.find_spec("torch") is not None
    has_openfold = importlib.util.find_spec("openfold") is not None
    data_dir = Path(__file__).parent / "data"
    has_data = data_dir.exists() and any(data_dir.iterdir())

    skip_torch = pytest.mark.skip(reason="PyTorch not available")
    skip_openfold = pytest.mark.skip(reason="OpenFold not available")
    skip_data = pytest.mark.skip(reason="Test data not available in tests/data/")

    for item in items:
        if "requires_torch" in item.keywords and not has_torch:
            item.add_marker(skip_torch)
        if "requires_openfold" in item.keywords and not has_openfold:
            item.add_marker(skip_openfold)
        if "requires_data" in item.keywords and not has_data:
            item.add_marker(skip_data)