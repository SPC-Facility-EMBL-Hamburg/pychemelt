"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np
import pytest

from pychemelt import Monomer as Sample
from pychemelt.utils.signals import signal_two_state_tc_unfolding
from pychemelt.utils.math import exponential_baseline

import tempfile

import os

def_params = {
    'DHm': 120,
    'Tm': 65+273.15,
    'Cp0': 1.8,
    'm0': 2.6,
    'm1': 0,
    'p1_N': 0,
    'p2_N': 100,
    'p3_N': 1,
    'p4_N': 0.1,
    'p1_U': 0,
    'p2_U': 110,
    'p3_U': 1,
    'p4_U': 0.2,
    'baseline_N_fx':exponential_baseline,
    'baseline_U_fx':exponential_baseline
}

def_concs = [0.1,0.5,1,2,3,4,5]

def aux_create_pychem_sim(params,concs):

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(20, 90, 70)
    temp_range_K = temp_range + 273.15

    signal_list = []
    temp_list   = []

    # Use a seeded Generator for reproducible noise in tests
    rng = np.random.default_rng(2)

    for D in concs:

        y = signal_two_state_tc_unfolding(temp_range_K, D, **params)

        y += rng.normal(0, 0.0005, len(y)) # Small error (seeded)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = Sample()

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo']   = [temp_range for _ in range(len(concs))]

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_denaturant_concentrations()

    pychem_sim.set_signal(['Fluo'])

    pychem_sim.select_conditions(normalise_to_global_max=False)
    pychem_sim.expand_multiple_signal()

    return pychem_sim

pychem_sim = aux_create_pychem_sim(def_params,def_concs)

def test_estimate_baseline_parameters_exponential():

    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='exponential',
        unfolded_baseline_type='exponential'
    )

    assert len(pychem_sim.third_param_Ns_per_signal[0]) == len(def_concs)

def test_fit_thermal_unfolding_local():

    pychem_sim.fit_thermal_unfolding_local()

    np.testing.assert_allclose(pychem_sim.Tms_multiple[0][0],65,rtol=0.3)

def test_guess_Cp():

    # Raise error if self.n_residues is zero
    pychem_sim.n_residues = 0
    with pytest.raises(ValueError):
        pychem_sim.guess_Cp()

    pychem_sim.n_residues = 130
    pychem_sim.guess_Cp()

    assert 1.4 <= pychem_sim.Cp0 <= 2.2 # 0.4 units tolerance from 1.8

    # Force exception clause so we just use the cp based on the number of residues
    pychem_sim.Tms_multiple = None
    pychem_sim.guess_Cp()

    assert 1.4 <= pychem_sim.Cp0 <= 2.2 # 0.3 units tolerance from 1.8


def test_fit_thermal_unfolding_global_err():

    pychem_sim.max_points = 200

    with pytest.raises(ValueError):
       pychem_sim.Cp0 = 0 # Force error
       pychem_sim.fit_thermal_unfolding_global()

def test_fit_thermal_unfolding_global():

    pychem_sim.Cp0 = 1.8
    pychem_sim.fit_thermal_unfolding_global()

    assert pychem_sim.params_df is not None

    expected = [65, 120, 1.8, 2.6]
    actual   = pychem_sim.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.3)

def test_estimate_derivative_prediction_interpolation():

    # Force non-evenly spaced temperatures to test interpolation in derivative estimation
    pychem_sim.temp_lst_expanded[0][0] -= 0.5

    pychem_sim.estimate_derivative()

    assert len(pychem_sim.deriv_lst_multiple[0]) == 7

    # Restore original temperatures for subsequent tests
    pychem_sim.temp_lst_expanded[0][0] += 0.5

    pychem_sim.fit_thermal_unfolding_global()


def test_fit_thermal_unfolding_global_global():

    pychem_sim.set_signal_id()
    pychem_sim.fit_thermal_unfolding_global_global()

    assert pychem_sim.params_df is not None

    expected = [65, 120, 1.8, 2.6]
    actual   = pychem_sim.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.3)

def test_fit_thermal_unfolding_global_global_global():

    pychem_sim.params_df = None

    pychem_sim.fit_thermal_unfolding_global_global_global()

    assert pychem_sim.params_df is not None

    expected = [65, 120, 1.8, 2.6]
    actual   = pychem_sim.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.3)

    df = pychem_sim.signal_to_df(signal_type='fitted',scaled=True)

    assert len(df) == 490

    df = pychem_sim.signal_to_df(signal_type='raw',scaled=True)

    assert len(df) == 490

    df = pychem_sim.signal_to_df(signal_type='fitted',scaled=False)

    assert len(df) == 490

def test_json_save_and_load():

    with tempfile.TemporaryDirectory() as tempdir:

        pychem_sim.save_state_as_json(os.path.join(tempdir, 'test.json'))

        sample2 = Sample()
        sample2.load_state_from_json(os.path.join(tempdir, 'test.json'))

        assert(type(pychem_sim.signal_dic['Fluo'][0][0]) == type(sample2.signal_dic['Fluo'][0][0]))
        assert(type(pychem_sim.signal_dic['Fluo'][0]) == type(sample2.signal_dic['Fluo'][0]))
        assert(type(pychem_sim.signal_dic['Fluo']) == type(sample2.signal_dic['Fluo']))

        assert(type(pychem_sim.temp_dic['Fluo'][0][0]) == type(sample2.temp_dic['Fluo'][0][0]))
        assert(type(pychem_sim.temp_dic['Fluo'][0]) == type(sample2.temp_dic['Fluo'][0]))
        assert(type(pychem_sim.temp_dic['Fluo']) == type(sample2.temp_dic['Fluo']))

        assert(type(pychem_sim.conditions) == type(sample2.conditions))
        assert(type(pychem_sim.labels) == type(sample2.labels))
        assert(type(pychem_sim.signals) == type(sample2.signals))

        assert(type(pychem_sim.first_param_Ns_per_signal) == type(sample2.first_param_Ns_per_signal))
        assert(type(pychem_sim.first_param_Ns_per_signal[0]) == type(sample2.first_param_Ns_per_signal[0]))

        assert(type(pychem_sim.t_melting_df_multiple[0]) == type(sample2.t_melting_df_multiple[0]))

        assert(type(pychem_sim.dg_df) == type(sample2.dg_df))

def test_compare_models():

    pychem_sim.compare_models(
        native_baseline_types=['exponential','constant'],
        unfolded_baseline_types=['exponential'],
        global_model_types=['global','global_global','global_global_global'])

    compare_df = pychem_sim.comparison_df

    # Check that the expected columns are present
    expected_columns = ['Native Baseline', 'Unfolded Baseline', 'Model Type', 'Tm', 'ΔHm', 'ΔCp', 'm-value', 'AIC']
    assert all(col in compare_df.columns for col in expected_columns)

    # Verify that the best model (lowest BIC) is the one with the correct baseline types and global model type
    best_model = compare_df.iloc[0]
    assert best_model['Native Baseline']   == 'exponential'
    assert best_model['Unfolded Baseline'] == 'exponential'
    assert best_model['Model Type'] == 'Global slopes and global intercepts'

    # Test it with additional arguments for the fit
    kwargs = {'dh_limits': [100,150], 'tm_limits': [60,70], 'cp_limits': [1.5,2.5]}
    pychem_sim.compare_models(
        native_baseline_types=['exponential','quadratic'],
            unfolded_baseline_types=['exponential'],
            global_model_types=['global','global_global','global_global_global'],
        **kwargs
    )

    assert pychem_sim.comparison_df is not None
    best_model = compare_df.iloc[0]
    assert best_model['Native Baseline']   == 'exponential'
    assert best_model['Unfolded Baseline'] == 'exponential'
    assert best_model['Model Type'] == 'Global slopes and global intercepts'

def test_compare_models_with_neff():

    pychem_sim.compare_models(
        native_baseline_types=['exponential'],
        unfolded_baseline_types=['exponential','linear'],
        global_model_types=['global','global_global','global_global_global'],
        neff=70
    )

    compare_df = pychem_sim.comparison_df

    # Check that the expected columns are present
    expected_columns = ['Native Baseline', 'Unfolded Baseline', 'Model Type', 'Tm', 'ΔHm', 'ΔCp', 'm-value', 'AIC']
    assert all(col in compare_df.columns for col in expected_columns)

    # Verify that the best model (lowest BIC) is the one with the correct baseline types and global model type
    best_model = compare_df.iloc[0]
    assert best_model['Native Baseline']   == 'exponential'
    assert best_model['Unfolded Baseline'] == 'exponential'
    assert best_model['Model Type'] == 'Global slopes and global intercepts'

    # Test it with additional arguments for the fit
    kwargs = {'dh_limits': [100,150], 'tm_limits': [60,70], 'cp_limits': [1.5,2.5]}
    pychem_sim.compare_models(
        native_baseline_types=['exponential'],
        unfolded_baseline_types=['exponential','quadratic'],
        global_model_types=['global','global_global','global_global_global'],
        neff=70,
        **kwargs
    )

    assert pychem_sim.comparison_df is not None
    best_model = compare_df.iloc[0]
    assert best_model['Native Baseline']   == 'exponential'
    assert best_model['Unfolded Baseline'] == 'exponential'
    assert best_model['Model Type'] == 'Global slopes and global intercepts'