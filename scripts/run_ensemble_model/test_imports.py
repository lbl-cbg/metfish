#!/usr/bin/env python
"""
Quick import test to verify train_structure.py dependencies work after cleanup
"""

import sys
import traceback

def test_imports():
    """Test that all required imports work"""
    print("Testing imports from train_structure.py...")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Basic imports
    try:
        import torch
        print("✓ torch imported successfully")
        tests.append(("torch", True, None))
    except Exception as e:
        print(f"✗ torch import failed: {e}")
        tests.append(("torch", False, str(e)))
    
    # Test 2: metfish.msa_model.config
    try:
        from metfish.msa_model.config import model_config
        print("✓ metfish.msa_model.config imported successfully")
        tests.append(("metfish.msa_model.config", True, None))
    except Exception as e:
        print(f"✗ metfish.msa_model.config import failed: {e}")
        tests.append(("metfish.msa_model.config", False, str(e)))
    
    # Test 3: metfish.msa_model.data.data_modules
    try:
        from metfish.msa_model.data.data_modules import MSASAXSDataset
        print("✓ MSASAXSDataset imported successfully")
        tests.append(("MSASAXSDataset", True, None))
    except Exception as e:
        print(f"✗ MSASAXSDataset import failed: {e}")
        tests.append(("MSASAXSDataset", False, str(e)))
    
    # Test 4: random_model.StructureModel
    try:
        from metfish.refinement_model.random_model import StructureModel
        print("✓ StructureModel imported successfully")
        tests.append(("StructureModel", True, None))
    except Exception as e:
        print(f"✗ StructureModel import failed: {e}")
        tests.append(("StructureModel", False, str(e)))
    
    # Test 5: random_model.compute_plddt_loss
    try:
        from metfish.refinement_model.random_model import compute_plddt_loss
        print("✓ compute_plddt_loss imported successfully")
        tests.append(("compute_plddt_loss", True, None))
    except Exception as e:
        print(f"✗ compute_plddt_loss import failed: {e}")
        tests.append(("compute_plddt_loss", False, str(e)))
    
    # Test 6: random_model.MSARandomModel (should exist for generate.py)
    try:
        from metfish.refinement_model.random_model import MSARandomModel
        print("✓ MSARandomModel imported successfully (needed by generate.py)")
        tests.append(("MSARandomModel", True, None))
    except Exception as e:
        print(f"✗ MSARandomModel import failed: {e}")
        tests.append(("MSARandomModel", False, str(e)))
    
    # Test 7: metfish.utils
    try:
        from metfish.utils import output_to_protein
        print("✓ metfish.utils.output_to_protein imported successfully")
        tests.append(("metfish.utils", True, None))
    except Exception as e:
        print(f"✗ metfish.utils import failed: {e}")
        tests.append(("metfish.utils", False, str(e)))
    
    # Test 8: openfold
    try:
        from openfold.np import protein
        print("✓ openfold.np.protein imported successfully")
        tests.append(("openfold.np.protein", True, None))
    except Exception as e:
        print(f"✗ openfold.np.protein import failed: {e}")
        tests.append(("openfold.np.protein", False, str(e)))
    
    print("=" * 60)
    
    # Summary
    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)
    
    print(f"\nTest Summary: {passed}/{total} imports successful")
    
    if passed == total:
        print("✓ All imports working! train_structure.py should run correctly.")
        return 0
    else:
        print("✗ Some imports failed. Check dependencies.")
        print("\nFailed imports:")
        for name, success, error in tests:
            if not success:
                print(f"  - {name}: {error}")
        return 1


def test_model_instantiation():
    """Test that StructureModel can be instantiated"""
    print("\n" + "=" * 60)
    print("Testing StructureModel instantiation...")
    print("=" * 60)
    
    try:
        from metfish.msa_model.config import model_config
        from metfish.refinement_model.random_model import StructureModel
        
        # Create config
        config = model_config('generating', train=True, low_prec=True)
        print("✓ Model config created")
        
        # Instantiate model
        model = StructureModel(config, training=True)
        print("✓ StructureModel instantiated successfully")
        
        # Check trainable parameters method exists
        trainable_params = model.get_trainable_parameters()
        print(f"✓ get_trainable_parameters() works (found {len(trainable_params)} params)")
        
        print("\n✓ StructureModel is functional!")
        return 0
        
    except Exception as e:
        print(f"✗ StructureModel instantiation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("IMPORT TEST FOR train_structure.py")
    print("After removing new_train.py and simple_test.py\n")
    
    # Run import tests
    import_result = test_imports()
    
    # Run model instantiation test
    model_result = test_model_instantiation()
    
    # Exit code
    exit_code = max(import_result, model_result)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("SUCCESS: All tests passed!")
        print("train_structure.py is ready to use.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILURE: Some tests failed.")
        print("Fix the issues above before running train_structure.py")
        print("=" * 60)
    
    sys.exit(exit_code)
