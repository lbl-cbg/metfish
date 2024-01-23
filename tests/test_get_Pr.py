import pandas as pd
import numpy.testing as nptest

from metfish.utils import get_Pr


def __run_test(path):
    r, p = get_Pr(path)
    df = pd.read_csv("tests/data/3nir.pr.csv")
    expected_r, expected_p = [df[k].values for k in ('r', 'P(r)')]

    nptest.assert_allclose(r, expected_r)
    nptest.assert_allclose(p, expected_p)


def test_cif():
    path = "tests/data/3nir.cif"
    __run_test(path)


def test_pdb():
    path = "tests/data/3nir.cif"
    __run_test(path)
