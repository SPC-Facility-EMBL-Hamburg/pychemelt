"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np

from pychemelt import Monomer as Sample
from pychemelt.utils.signals import signal_two_state_tc_unfolding

from pychemelt.utils.math import quadratic_baseline
import pytest

KCAL_TO_KJ_CST = 4.184

def_params = { 
    'DHm': 100,
    'Tm': 60 + 273.15,
    'Cp0': 1.6,
    'm0': 2.6,
    'm1': 0,
    'p1_N': -0.1,
    'p2_N': 1.5,
    'p3_N': -0.015,  # Negative temperature dependence for native state
    'p4_N': 0.0001,
    'p1_U': -0.005,
    'p2_U': 2.5,
    'p3_U': -0.025,  # Negative temperature dependence for unfolded state
    'p4_U': 0.0002,
    'baseline_N_fx':quadratic_baseline,
    'baseline_U_fx':quadratic_baseline
}

def_concs = [1e-8,1,1.5,2,2.6,3,4,5]

scalings_factors = np.array([1,0.95,1,1.1,1,1,1,0.96])

def aux_create_pychem_sim(params,concs,signal_error=0.0005):

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(20, 81, 61)
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

        temp_range[-1] = np.nan # To test that the code can handle NaN values in the temperature array

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
    pychem_sim.set_units('international')
    pychem_sim.select_conditions(normalise_to_global_max=False)
    pychem_sim.set_temperature_range(5, 100)

    pychem_sim.expand_multiple_signal()

    return pychem_sim

# --------- #  Create global pychem_sim object for the rest of tests  # --------- #
sample = aux_create_pychem_sim(def_params,def_concs)
sample.estimate_derivative()
sample.guess_Tm()
sample.n_residues = 130

sample.set_units('international')

sample.estimate_baseline_parameters(
    native_baseline_type='quadratic',
    unfolded_baseline_type='quadratic',
    window_range_native=16,
    window_range_unfolded=16
)
sample.fit_thermal_unfolding_local()
sample.guess_Cp()

sample.guess_initial_parameters(
    native_baseline_type='quadratic',
    unfolded_baseline_type='quadratic',
    window_range_native=16,
    window_range_unfolded=16
)


def test_fit_thermal_unfolding_global():

    sample.set_thermodynamic_params_guess()

    p0 = sample.p0_thermodynamics.copy()

    sample.set_thermodynamic_params_guess(user_thermodynamic_params_guess=p0)

    sample.fit_thermal_unfolding_global(set_init_params = False)

    assert sample.params_df is not None

    assert sample.params_df.shape[0] == 52

    expected = [60+273.15, 100*KCAL_TO_KJ_CST, 1.6*KCAL_TO_KJ_CST, 2.6*KCAL_TO_KJ_CST]
    actual   = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

    args_dic = {
        'fit_m_dep': True,
        'dh_limits': [50 * KCAL_TO_KJ_CST, 200 * KCAL_TO_KJ_CST],
        'tm_limits': [40 + 273.15, 80 + 273.15],
        'cp_limits': [0.5 * KCAL_TO_KJ_CST, 4 * KCAL_TO_KJ_CST]
    }

    for key,val in args_dic.items():

        sample.fit_thermal_unfolding_global(**{key:val})
        actual = sample.params_df.iloc[:4,1]
        np.testing.assert_allclose(actual,expected,rtol=0.1)

    # -- Fit with fixed Cp -- #
    sample.fit_thermal_unfolding_global(cp_value=1.6 * KCAL_TO_KJ_CST)

    expected.pop(2)
    actual = sample.params_df.iloc[:3,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_fit_thermal_unfolding_global_global():

    sample.global_fit_done = False # Force re-fitting

    sample.fit_thermal_unfolding_global_global()

    expected = [60+273.15, 100*KCAL_TO_KJ_CST, 1.6*KCAL_TO_KJ_CST, 2.6*KCAL_TO_KJ_CST]
    actual = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_fit_thermal_unfolding_global_global_global():

    sample.fit_thermal_unfolding_global() # Needs to be done firsts
    sample.global_global_fit_done = False # Force re-fitting clause
    sample.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    expected = [60+273.15, 100*KCAL_TO_KJ_CST, 1.6*KCAL_TO_KJ_CST, 2.6*KCAL_TO_KJ_CST]
    actual = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_create_confidence_intervals():

    expected = [60+273.15, 100*KCAL_TO_KJ_CST, 1.6*KCAL_TO_KJ_CST, 2.6*KCAL_TO_KJ_CST]

    for percentage in [0.68, 0.95, 0.99]:

        sample.calculate_confidence_intervals(percentage=percentage)

        # Now verify that each TRUE parameter value is within the confidence interval
        for i, value in enumerate(expected):

            lower_bound = sample.ci_df.iloc[i,1]  # Assuming the second column is the lower CI
            upper_bound = sample.ci_df.iloc[i,3]  # Assuming the fourth column is the upper CI

            assert lower_bound <= value <= upper_bound, f"Parameter {sample.ci_df.iloc[i,0]}: {value} not in [{lower_bound}, {upper_bound}] for percentage={percentage}"

def test_leave_one_out_cross_validation():

    expected = [60+273.15, 100*KCAL_TO_KJ_CST, 1.6*KCAL_TO_KJ_CST, 2.6*KCAL_TO_KJ_CST]

    sample.leave_one_out_cross_validation()

    assert sample.loo_df is not None
    
    # Verify that each TRUE parameter value is within the confidence interval of the LOO CV results
    for i, value in enumerate(expected):

        # Second column is LOO mean, third column is LOO std
        loo_mean = sample.loo_df.iloc[i,1]
        loo_std = sample.loo_df.iloc[i,2]
        lower_bound = (loo_mean - 2*loo_std)*0.9999
        upper_bound = (loo_mean + 2*loo_std)

        assert lower_bound <= value <= upper_bound, f"Parameter {sample.loo_df.iloc[i,0]}: {value} not in [{lower_bound}, {upper_bound}] for LOO CV"

        # Verify parameter names
        assert sample.loo_df.iloc[:,0].to_list() == ['Tm (K)', 'ΔHm (kJ / mol)', 'ΔCp (kJ / mol / K)', 'm-value (kJ / mol / M)']

def test_signal_to_df():

    signal_type_options = ['raw','derivative']

    for signal_type in signal_type_options:

        df = sample.signal_to_df(signal_type=signal_type, scaled=False)

        assert len(df) == 480

    signal_type_options = ['raw','fitted']

    for signal_type in signal_type_options:

        df = sample.signal_to_df(signal_type=signal_type, scaled=True)

        assert len(df) == 480
        assert np.max(df['Signal']) <= 100

# Now we need to test LOO when the CI can not be calculated 
def test_loo_no_ci():
    def_params['Cp0'] = 0 # To force raise error in CI calculation
    sample = aux_create_pychem_sim(def_params,def_concs[:6])
    sample.estimate_derivative()
    sample.guess_Tm()
    sample.n_residues = 130
    sample.estimate_baseline_parameters('quadratic','quadratic')
    sample.fit_thermal_unfolding_local()
    sample.guess_Cp()
    sample.fit_thermal_unfolding_global()

    with pytest.raises(ValueError):
        sample.leave_one_out_cross_validation() 

def test_create_fit_report():

    sample.create_fit_report()

    assert isinstance(sample.fit_report, str)

def test_create_fit_report_with_neff():
    
    sample.fit_report = None # Reset fit report to force re-creation with neff
    
    sample.create_fit_report(neff=20)

    assert isinstance(sample.fit_report, str)

def test_create_fit_report_err():

    sample.result = None # Force error
    with pytest.raises(ValueError):
        sample.create_fit_report()