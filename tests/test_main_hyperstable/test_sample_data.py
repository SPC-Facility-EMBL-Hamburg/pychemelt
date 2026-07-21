"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np

from pychemelt import Monomer as Sample
from pychemelt.utils.signals import signal_two_state_tc_unfolding

from pychemelt.utils.math import linear_baseline
import pytest

DHm_VAL = 76.4
Tm_VAL = 145
Cp0_VAL = 0.7
M_VAL = 1.5

def_params = { 
    'DHm': DHm_VAL,
    'Tm': Tm_VAL + 273.15,
    'Cp0': Cp0_VAL,
    'm0': M_VAL,
    'm1': 0,
    'p1_N': 0,
    'p2_N': 99.46,
    'p3_N': -0.844,
    'p4_N': -0.5,
    
    'p1_U': 0,
    'p2_U': 0.03,
    'p3_U': 0.156,  
    'p4_U': 0.67,
    'baseline_N_fx':linear_baseline,
    'baseline_U_fx':linear_baseline
}

def_concs = [2,2.5,3.5,4.5,5.5,6]

scalings_factors = np.array([1,0.95,1,1.1,1,1])

def aux_create_pychem_sim(params,concs,signal_error=0.0005):

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(20, 95, 61)
    temp_range_K = temp_range + 273.15
    signal_list = []
    temp_list   = []

    for i,D in enumerate(concs):

        y = signal_two_state_tc_unfolding(temp_range_K, D, **params)

        rng = np.random.default_rng(2)

        # Add gaussian error to signal
        y += rng.normal(0, signal_error, len(y)) # Small error

        # Add gaussian error to PROTEIN concentration
        y *= scalings_factors[i]

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = Sample()

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo']   = [temp_range for _ in range(len(concs))]

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_denaturant_concentrations()

    pychem_sim.set_signal('Fluo')

    pychem_sim.select_conditions(normalise_to_global_max=False)
    pychem_sim.expand_multiple_signal()

    return pychem_sim



# --------- #  Create global pychem_sim object for the rest of tests  # --------- #
sample = aux_create_pychem_sim(def_params,def_concs)
sample.estimate_derivative()
sample.guess_Tm()
sample.n_residues = 130

def test_global_global_fit():

    sample.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='linear',
        window_range_native=16,
        window_range_unfolded=16
    )

    sample.fit_thermal_unfolding_local()
    sample.guess_Cp()
    sample.guess_initial_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='linear',
        window_range_native=16,
        window_range_unfolded=16
    )
    sample.fit_thermal_unfolding_global()
    sample.fit_thermal_unfolding_global_global()
    
    expected = [Tm_VAL, DHm_VAL, Cp0_VAL, M_VAL]
    actual   = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.3)

def test_global_global_global_fit():

    sample.fit_thermal_unfolding_global_global_global()
    expected = [Tm_VAL, DHm_VAL, Cp0_VAL, M_VAL]
    actual   = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.3)
