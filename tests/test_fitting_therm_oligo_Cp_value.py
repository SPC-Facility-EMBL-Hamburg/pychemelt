"""
Module for testing fitting of thermal unfolding curves with different Cp values.
"""

"""
Module for fitting of thermal unfolding curves.

Tests the fitting methods used by ThermalOligomer for their capabilities of fitting to simulated data
"""

import numpy as np

from pychemelt.utils.fitting import (
    fit_oligomer_unfolding_single_slopes,
    fit_oligomer_unfolding_three_states_single_slopes
)

from pychemelt.utils.math import constant_baseline_only_temp as constant_baseline

from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx,
    map_three_state_model_to_signal_fx
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 30.0
TEMP_STOP = 90.0
N_TEMPS = 80
CONCS = np.array([2,6,24,72])*1e-6

# Two state
# Model / ground-truth parameters
DHm_VAL = 250
Tm_VAL = 70
CP0_VAL = 1.8


INTERCEPT_N = 50
INTERCEPT_U = 100

rng = np.random.default_rng(RNG_SEED)

def_params = {
    'dHm': DHm_VAL,
    'Tm': Tm_VAL+273.15,
    'Cp': CP0_VAL,
    'p1_N': INTERCEPT_N,
    'p2_N': 0,
    'p3_N': 0,
    'p1_U': INTERCEPT_U,
    'p2_U': 0,
    'p3_U': 0,
    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline,
}

# Three state
# Model / ground-truth parameters
DHm_VAL_1 = 300
DHm_VAL_2 = 300
Tm_VAL_1 = 50
Tm_VAL_2 = 70
CP1_VAL = 1.0
CPTH_VAL = 2.0


INTERCEPT_I = 100

INTERCEPT_N = 80
INTERCEPT_U = 110


def_params_three_state = {
    'DH1': DHm_VAL_1,
    'DH2': DHm_VAL_2,
    'T1': Tm_VAL_1+273.15,
    'T2': Tm_VAL_2+273.15,
    'bI': INTERCEPT_I,
    'p1_N': INTERCEPT_N,
    'p2_N': 0,
    'p3_N': 0,
    'p1_U': INTERCEPT_U,
    'p2_U': 0,
    'p3_U': 0,
    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline,
    "Cp1": 0,
    'CpTh': 0,
}


concs = CONCS

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
temp_range_K = temp_range + 273.15

def fitting_data_two_state(oligomer, cp_true, cp_fit, params=def_params):
    signal_fx = map_two_state_model_to_signal_fx(oligomer)

    signal_list = []
    temp_list = []

    params['Cp'] = cp_true

    for C in concs:
        y = signal_fx(temp_range_K, C, **params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL - 100] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL + 100] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'cp_value': cp_fit,
    }

    global_fit_params, cov, predicted_lst, _, _ = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    return global_fit_params, cov, predicted_lst

def fitting_data_three_state(oligomer, cp1_true, cpth_true, cpth_fit, cp1_fit=1.0, params=def_params_three_state):
    signal_fx = map_three_state_model_to_signal_fx(oligomer)

    signal_list = []
    temp_list = []

    params['Cp1'] = cp1_true
    params['CpTh'] = cpth_true

    for C in concs:
        y = signal_fx(temp_range_K, C, **params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    if cpth_fit is None:
        p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
            INTERCEPT_I] * len(concs)
        low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
        high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))
    else:
        p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, cp1_fit] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
            INTERCEPT_I] * len(concs)
        low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100, 0.1] + [1e-5] * (3 * len(concs))
        high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100, cpth_fit] + [1e3] * (3 * len(concs))


    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'CpTh_value': cpth_fit,
    }

    global_fit_params, cov, predicted_lst, _, _ = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    return global_fit_params, cov, predicted_lst

def test_fit_monomer_unfolding_single_slopes_Cp_values():

    oligomer = "Monomer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)


    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)


def test_fit_dimer_unfolding_single_slopes_Cp_values():
    oligomer = "Dimer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)


def test_fit_trimer_unfolding_single_slopes_Cp_values():
    oligomer = "Trimer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)

def test_fit_tetramer_unfolding_single_slopes_Cp_values():
    oligomer = "Tetramer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)

# Testing three state fitting

def test_fit_monomer_unfolding_three_states_single_slopes_Cp_values():

    oligomer = "Monomer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_dimer_unfolding_three_states_single_slopes_Cp_values():
    oligomer = "Dimer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_trimer_unfolding_three_states_single_slopes_Cp_values():

    oligomer = "Trimer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

def test_fit_tetramer_unfolding_three_states_single_slopes_Cp_values():
    oligomer = "Tetramer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_dimer_dimeric_unfolding_three_states_single_slopes_Cp_values():
    oligomer = "Dimer_dimeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_trimer_trimeric_unfolding_three_states_single_slopes_Cp_values():

    oligomer = "Trimer_trimeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)
