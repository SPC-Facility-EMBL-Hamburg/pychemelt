"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np

from pychemelt import Monomer as Sample
from pychemelt.utils.signals import signal_two_state_tc_unfolding

from pychemelt.utils.math import constant_baseline
import pytest

DHm = 120
Tm = 60 
Cp0 = 1.6
m0 = 3.2

def_params_signal_A = {
    'DHm': DHm,
    'Tm': Tm + 273.15,
    'Cp0': Cp0,
    'm0': m0,
    'm1': 0,
    'p1_N': -0.1,
    'p2_N': 1.5,
    'p3_N': -0.015,  # Negative temperature dependence for native state
    'p4_N': 0.0001,
    'p1_U': -0.005,
    'p2_U': 2.5,
    'p3_U': -0.025,  # Negative temperature dependence for unfolded state
    'p4_U': 0.0002,
    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline
}

def_params_signal_B = {
    'DHm': DHm,
    'Tm': Tm + 273.15,
    'Cp0': Cp0,
    'm0': m0,
    'm1': 0,
    'p1_N': -0.1,
    'p2_N': 1.5,
    'p3_N': -0.015,  # Negative temperature dependence for native state
    'p4_N': 0.0001,
    'p1_U': -0.005,
    'p2_U': 2.5,
    'p3_U': -0.025,  # Negative temperature dependence for unfolded state
    'p4_U': 0.0002,
    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline
}

EXPECTED = [Tm, DHm, Cp0, m0]
TOLERANCES = [x/20 for x in EXPECTED] # Five percent tolerance for all parameters

concs = [0.1*(1.5**i) for i in range(10)] 

rng = np.random.default_rng(2)

scalings_factors = rng.uniform(low=0.95, high=1.05, size=len(concs))

# Calculate signal range for proper y-axis scaling
temp_range = np.linspace(20, 80, 60)
temp_range_K = temp_range + 273.15
signal_list_A = []
signal_list_B = []
temp_list   = []

for i,D in enumerate(concs):

    y_A = signal_two_state_tc_unfolding(temp_range_K, D, **def_params_signal_A)
    y_B = signal_two_state_tc_unfolding(temp_range_K, D, **def_params_signal_B)

    # Add gaussian error to signal
    y_A += rng.normal(0, 0.025, len(y_A)) # Small error
    y_B += rng.normal(0, 0.025, len(y_B)) # Small error

    # Add gaussian error to PROTEIN concentration
    y_A *= scalings_factors[i]
    y_B *= scalings_factors[i]

    signal_list_A.append(y_A)
    signal_list_B.append(y_B)

    temp_list.append(temp_range)

pychem_sim = Sample()

pychem_sim.signal_dic['FluoA'] = signal_list_A
pychem_sim.signal_dic['FluoB'] = signal_list_B
pychem_sim.temp_dic['FluoA']   = [temp_range for _ in range(len(concs))]
pychem_sim.temp_dic['FluoB']   = [temp_range for _ in range(len(concs))]

pychem_sim.conditions = concs

pychem_sim.global_min_temp = np.min(temp_range)
pychem_sim.global_max_temp = np.max(temp_range)

pychem_sim.set_denaturant_concentrations()

pychem_sim.set_signal(['FluoA','FluoB'])   

pychem_sim.select_conditions(normalise_to_global_max=False)
pychem_sim.expand_multiple_signal()
pychem_sim.n_residues = 130

def intervals_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)

def test_leave_one_out():
    
    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='constant',
        unfolded_baseline_type='constant'
    )

    pychem_sim.fit_thermal_unfolding_local()
    pychem_sim.guess_Cp()

    for function in ['fit_thermal_unfolding_global', 'fit_thermal_unfolding_global_global', 'fit_thermal_unfolding_global_global_global']:
        
        getattr(pychem_sim, function)()

        pychem_sim.leave_one_out_cross_validation()

        assert pychem_sim.loo_df is not None
        
        # Verify that each TRUE parameter value is within the confidence interval of the LOO CV results
        for i, value in enumerate(EXPECTED):

            # Second column is LOO mean, third column is LOO std
            loo_mean = pychem_sim.loo_df.iloc[i,1]
            loo_std = pychem_sim.loo_df.iloc[i,2]
            lower_bound = (loo_mean - 2*loo_std)
            upper_bound = (loo_mean + 2*loo_std)

            # Find if intervals overalap, based on expected and tolerance
            tol = TOLERANCES[i]
            assert intervals_overlap(lower_bound, upper_bound, value - tol, value + tol), f"Parameter {pychem_sim.loo_df.iloc[i,0]}: {value} not in [{lower_bound}, {upper_bound}] for LOO CV"
