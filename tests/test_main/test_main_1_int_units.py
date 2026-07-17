"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np
import pytest

from pychemelt import Monomer as Sample

sample = Sample()


def test_preprocess_with_SI_units():

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()
    sample.set_signal(['350nm'])
    sample.set_units('international')

    # Select with scaling
    sample.select_conditions(
        [False for _ in range(24)] + [True for _ in range(8)] + [False for _ in range(16)],
        normalise_to_global_max=True
    )

    assert np.min(sample.temp_lst_multiple) > 273.15

    # Raise error if t_max < t_min
    with pytest.raises(ValueError):
        sample.set_temperature_range(80, 30)

    sample.set_temperature_range(30, 80)

    assert np.min(sample.temp_lst_multiple[0]) >= 30

    sample.set_temperature_range(5, 100)

    assert sample.user_min_temp == 5+273.15
    assert sample.user_max_temp == 100+273.15
