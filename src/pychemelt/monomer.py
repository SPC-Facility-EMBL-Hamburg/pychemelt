"""
Main class to handle thermal and chemical denaturation data
The current model assumes the protein is a monomer and that the unfolding is reversible
"""

import pandas as pd
import numpy as np

from itertools import chain
from copy import deepcopy
from lmfit import fit_report

from .main import Sample

from .utils.signals import signal_two_state_tc_unfolding

from .utils.math import (
    temperature_to_celsius,
    temperature_to_kelvin,
    relative_errors,
    find_line_outliers,
    aic_bic_eff,
    extended_bic,
    shift_temperature
)

from .utils.processing import (
    fit_local_thermal_unfolding_to_signal_lst,
    set_param_bounds,
    adjust_value_to_interval,
    re_arrange_params,
    re_arrange_predictions,
    subset_data,
    transform_to_list,
    ci_dict_to_summary_df,
    re_arrange_loo_initial_params,
    find_baseline_params
)

from .utils.fitting import (
    fit_line_robust,
    fit_tc_unfolding_single_slopes,
    fit_tc_unfolding_shared_slopes_many_signals,
    fit_tc_unfolding_many_signals,
    evaluate_fitting_and_refit,
    baseline_fx_name_to_req_params,
    compute_asymmetric_confidence_intervals
)

class Monomer(Sample):
    """
    Class to hold the data of a single sample and fit it
    """

    def __init__(self, name='Test'):

        super().__init__(name)

        self.fit_m_dep = False  # Fit the temperature dependence of the m-value
        self.thermodynamic_params_guess = None
        self.nr_den = 0  # Number of denaturant concentrations
        self.oligomeric = False # Flag for oligomer for plotting
        self.model_scale_factor = False # Flag for model scale factor for fitting
        self.Tms = None 

    def set_denaturant_concentrations(self, concentrations=None):

        """
        Set the denaturant concentrations for the sample

        Parameters
        ----------
        concentrations : list, optional
            List of denaturant concentrations. If None, use the sample conditions

        Notes
        -----
        Creates/updates attribute `denaturant_concentrations_pre` (numpy.ndarray)
        """

        if concentrations is None:
            concentrations = self.conditions

        concentrations = transform_to_list(concentrations)

        self.denaturant_concentrations_pre = np.array(concentrations)

        return None

    def select_conditions(
            self,
            boolean_lst=None,
            normalise_to_global_max=True):

        """
        For each signal, select the conditions to be used for the analysis

        Parameters
        ----------
        boolean_lst : list of bool, optional
            List of booleans selecting which conditions to keep. If None, keep all.
        normalise_to_global_max : bool, optional
            If True, normalise the signal to the global maximum - per signal type

        Notes
        -----
        Creates/updates several attributes used by downstream fitting:
        - signal_lst_multiple, temp_lst_multiple : lists of lists with selected data
        - denaturant_concentrations : list of selected denaturant concentrations
        - denaturant_concentrations_expanded : flattened numpy array matching expanded signals
        - boolean_lst, normalise_to_global_max, nr_den : control flags/values
        """

        # If boolean_lst is a boolean, convert it to a list of one boolean
        boolean_lst = transform_to_list(boolean_lst)

        if boolean_lst is None:
            self.signal_lst_multiple = self.signal_lst_pre_multiple
            self.temp_lst_multiple = self.temp_lst_pre_multiple
            self.denaturant_concentrations = self.denaturant_concentrations_pre
        else:

            self.signal_lst_multiple = [None for _ in range(len(self.signal_lst_pre_multiple))]
            self.temp_lst_multiple = [None for _ in range(len(self.temp_lst_pre_multiple))]

            for i in range(len(self.signal_lst_pre_multiple)):
                self.signal_lst_multiple[i] = [x for j, x in enumerate(self.signal_lst_pre_multiple[i]) if
                                               boolean_lst[j]]
                self.temp_lst_multiple[i] = [x for j, x in enumerate(self.temp_lst_pre_multiple[i]) if boolean_lst[j]]

            self.denaturant_concentrations = [x for i, x in enumerate(self.denaturant_concentrations_pre) if
                                              boolean_lst[i]]

        if normalise_to_global_max:

            flat = list(chain.from_iterable(chain.from_iterable(self.signal_lst_multiple)))
            global_max = np.nanmax(flat)  # Global maximum across all signals

            for i in range(len(self.signal_lst_multiple)):
                self.signal_lst_multiple[i] = [x / global_max * 100 for x in self.signal_lst_multiple[i]]

        self.nr_den = len(self.denaturant_concentrations)

        # Expand the number of denaturant concentrations to match the number of signals
        denaturant_concentrations = [self.denaturant_concentrations for _ in range(self.nr_signals)]

        self.denaturant_concentrations_expanded = np.concatenate(denaturant_concentrations, axis=0)

        self.boolean_lst = boolean_lst
        self.normalise_to_global_max = normalise_to_global_max

        self.denaturant_concentrations = np.array(self.denaturant_concentrations)

        return None


    def fit_thermal_unfolding_local(self):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit one curve at a time, with individual parameters
        """

        # Require self.t_melting_init_multiple
        if self.t_melting_init_multiple is None:

            self.estimate_derivative()
            self.guess_Tm()

        self.Tms_multiple = []
        self.dHs_multiple = []
        self.predicted_lst_multiple = []

        for i in range(len(self.signal_lst_multiple)):

            Tms, dHs, predicted_lst = fit_local_thermal_unfolding_to_signal_lst(
                self.signal_lst_multiple[i],
                self.temp_lst_multiple[i],
                self.t_melting_init_multiple[i],
                self.first_param_Ns_per_signal[i],
                self.first_param_Us_per_signal[i],
                self.second_param_Ns_per_signal[i],
                self.second_param_Us_per_signal[i],
                self.third_param_Ns_per_signal[i],
                self.third_param_Us_per_signal[i],
                baseline_native_fx=self.baseline_N_fx,
                baseline_unfolded_fx=self.baseline_U_fx
            )

            self.Tms_multiple.append(Tms)
            self.dHs_multiple.append(dHs)
            self.predicted_lst_multiple.append(predicted_lst)

        self.single_fit_done = True

        return None

    def guess_Cp(self):

        """
        Guess the Cp of the sample by fitting a line to the Tm and dH values

        Notes
        -----
        This method creates/updates attributes used later in fitting:
        - Tms, dHs, slope_dh_tm, intercept_dh_tm, Cp0, Cp0 assigned to self.Cp0
        """

        # If the number of residues is still zero, raise an error
        if self.n_residues == 0:
            raise ValueError('The number of residues is still zero. Please set n_residues before calling guess_Cp')

        # Requires self.single_fit_done

        expected_Cp0 = self.n_residues * 0.0148 - 0.1267

        if not self.single_fit_done:
            self.fit_thermal_unfolding_local()

        try:

            Tms = []
            dHs = []

            for i in range(len(self.Tms_multiple)):
                Tms.extend(self.Tms_multiple[i])
                dHs.extend(self.dHs_multiple[i])

            self.Tms = Tms
            self.dHs = dHs

            tms = np.array(self.Tms)
            dhs = np.array(self.dHs)

            m, b = fit_line_robust(tms, dhs)

            outliers = find_line_outliers(m, b, tms, dhs)

            if len(outliers) > 0:
                # Remove outliers
                tms = np.delete(tms, outliers)
                dhs = np.delete(dhs, outliers)

                # Assign the new values
                self.Tms = tms
                self.dHs = dhs

                m, b = fit_line_robust(self.Tms, self.dHs)

            self.slope_dh_tm = m
            self.intercept_dh_tm = b

            Cp0 = m if m > 0 else -1

            expected_lower_Cp = 0.3 if np.max(self.Tms) > 100 else expected_Cp0 / 1.6

            # Verify that the initial Cp is between the expected range
            if Cp0 < expected_lower_Cp or Cp0 > expected_Cp0 * 1.5:
                Cp0 = expected_Cp0

        except:

            Cp0 = expected_Cp0

        # Cp0 needs to be positive
        Cp0 = max(Cp0, 0.3)

        self.Cp0 = Cp0

        return None

    def guess_initial_parameters(
            self,
            native_baseline_type,
            unfolded_baseline_type,
            window_range_native=12,
            window_range_unfolded=12
    ):
        """
        Estimate starting thermodynamic and baseline parameters for global fitting.

        Parameters
        ----------
        native_baseline_type : {'constant', 'linear', 'quadratic', 'exponential'}
            The model type for the native state baseline.
        unfolded_baseline_type : {'constant', 'linear', 'quadratic', 'exponential'}
            The model type for the unfolded state baseline.
        window_range_native : float, optional
            Temperature range at the start of the curve (in degrees) used for
            native baseline estimation. Default is 12.
        window_range_unfolded : float, optional
            Temperature range at the end of the curve used for unfolded
            baseline estimation. Default is 12.
        """

        # We will use the Ratio signal, if available, to estimate the initial parameters
        use_ratio = 'Ratio' in self.signals and 'Ratio' not in self.signal_names

        if use_ratio:

            current_signals = self.signal_names

            # Extract temperature limits
            self.set_signal('Ratio')
            self.select_conditions(self.boolean_lst, normalise_to_global_max=True)
            self.set_temperature_range(self.user_min_temp, self.user_max_temp)

        # Fit the data using the linear - constant option
        self.estimate_baseline_parameters(
            native_baseline_type,
            unfolded_baseline_type,
            window_range_native,
            window_range_unfolded
        )

        self.fit_thermal_unfolding_local()
        self.guess_Cp()

        # Apply a first fitting round to obtain initial estimates for the thermodynamic parameters
        self.fit_thermal_unfolding_global(predict_baselines=False)

        self.thermodynamic_params_guess = self.global_fit_params[:4]

        if use_ratio:
            # Go back to the original signal
            self.set_signal(current_signals)
            self.select_conditions(self.boolean_lst, normalise_to_global_max = self.normalise_to_global_max)
            self.set_temperature_range(self.user_min_temp, self.user_max_temp)

        return None

    def create_dg_df(self):

        """
        Create a dataframe of the dg values versus temperature
        """

        # Create a dataframe of the parameters
        Tm, DHm, Cp0 = self.global_fit_params[:3]

        T_c = np.arange(0, 150, 0.5)
        T = temperature_to_kelvin(T_c)
        Tm = temperature_to_kelvin(Tm)

        DG = DHm * (1 - T / Tm) + Cp0 * (T - Tm - T * np.log(T / Tm))

        dg_df = pd.DataFrame({
            'DG (kcal/mol)': DG,
            'Temperature (°C)': T_c
        })

        self.dg_df = dg_df

        return None

    def get_current_thermodynamic_params_guess(self):

        """
        Get the current guess for the thermodynamic parameters (Tm, dH, Cp, m-value)
        
        Returns
        -------
        list
            List of four values, the current guess for the thermodynamic parameters (Tm, dH, Cp, m-value)
        """

        # Raise an error if self.Tms is None or empty
        if self.Tms is None or len(self.Tms) == 0:
            raise ValueError('Tms is None or empty. Please run guess_Cp before calling this method.')

        max_tm_id = np.argmax(self.Tms)

        if self.thermodynamic_params_guess is None:

            p0 = [self.Tms[max_tm_id], np.max([self.dHs[max_tm_id], 80]), self.Cp0, 2.8]

        else:

            p0 = self.thermodynamic_params_guess

        return p0

    def fit_thermal_unfolding_global(
            self,
            fit_m_dep=False,
            cp_limits=None,
            dh_limits=None,
            tm_limits=None,
            cp_value=None,
            predict_baselines=True,
            user_thermodynamic_params_guess=None):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters but local slopes and local baselines)
        Multiple signals can be fitted at the same time, such as 350nm and 330nm

        Parameters
        ----------
        fit_m_dep : bool, optional
            If True, fit the temperature dependence of the m-value
        cp_limits : list, optional
            List of two values, the lower and upper bounds for the Cp value. If None, bounds set automatically
        dh_limits : list, optional
            List of two values, the lower and upper bounds for the dH value. If None, bounds set automatically
        tm_limits : list, optional
            List of two values, the lower and upper bounds for the Tm value. If None, bounds set automatically
        cp_value : float, optional
            If provided, the Cp value is fixed to this value, the bounds are ignored
        predict_baselines : bool, optional
            If True, predict the baselines after fitting and store them in the object. Default is True.
        thermodynamic_params_guess : list, optional
            List of four values, the initial guess for the thermodynamic parameters (Tm, dH, Cp, m-value). If None, initial guess set automatically
        Notes
        -----
        This is a heavy routine that creates/updates many fitting-related attributes, including:
        - bNs_expanded, bUs_expanded, kNs_expanded, kUs_expanded, qNs_expanded, qUs_expanded
        - p0, low_bounds, high_bounds, global_fit_params, rel_errors
        - predicted_lst_multiple, params_names, params_df, dg_df
        - flags: global_fit_done, fit_m_dep, limited_tm, limited_dh, limited_cp, fixed_cp
        """

        self.global_global_global_fit_done = False  # Reset the flag for the more complex global fit
        self.global_global_fit_done = False  # Reset the flag for the global fit with shared slopes

        # Requires Cp0
        if self.Cp0 <= 0:
            raise ValueError('Cp0 must be positive. Please run guess_Cp before fitting globally.')

        if user_thermodynamic_params_guess is not None:

            p0 = user_thermodynamic_params_guess
        
        else:

            p0 = self.get_current_thermodynamic_params_guess()

        params_names = [
            'Tm (°C)',
            'ΔHm (kcal/mol)',
            'Cp (kcal/mol/°C)',
            'm-value (kcal/mol/M)']

        self.first_param_Ns_expanded = np.concatenate(self.first_param_Ns_per_signal, axis=0)
        self.first_param_Us_expanded = np.concatenate(self.first_param_Us_per_signal, axis=0)
        self.second_param_Ns_expanded = np.concatenate(self.second_param_Ns_per_signal, axis=0)
        self.second_param_Us_expanded = np.concatenate(self.second_param_Us_per_signal, axis=0)
        self.third_param_Ns_expanded = np.concatenate(self.third_param_Ns_per_signal, axis=0)
        self.third_param_Us_expanded = np.concatenate(self.third_param_Us_per_signal, axis=0)

        p0 = np.concatenate([p0, self.first_param_Ns_expanded, self.first_param_Us_expanded])

        # We need to append as many bN and bU as the number of denaturant concentrations
        # times the number of signal types
        for signal in self.signal_names:

            params_names += (['intercept_native - ' + str(self.denaturant_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_den)])

        for signal in self.signal_names:

            params_names += (['intercept_unfolded - ' + str(self.denaturant_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_den)])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p0 = np.concatenate([p0, self.second_param_Ns_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p0 = np.concatenate([p0, self.second_param_Us_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p0 = np.concatenate([p0, self.third_param_Ns_expanded])
            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p0 = np.concatenate([p0, self.third_param_Us_expanded])

            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        low_bounds = (p0.copy())
        high_bounds = (p0.copy())

        low_bounds[4:], high_bounds[4:] = set_param_bounds(p0[4:],params_names[4:])

        self.limited_tm = tm_limits is not None

        if self.limited_tm:

            tm_lower, tm_upper = tm_limits

        else:

            tm_lower = p0[0] - 12
            tm_upper = np.max([self.user_max_temp + 20, p0[0] + 10])

        low_bounds[0] = tm_lower
        high_bounds[0] = tm_upper

        # Verify that the initial guess is within the user-defined limits
        p0[0] = adjust_value_to_interval(p0[0], tm_lower, tm_upper,1)

        self.limited_dh = dh_limits is not None

        if self.limited_dh:

            dh_lower, dh_upper = dh_limits

            p0[1] = adjust_value_to_interval(p0[1], dh_lower, dh_upper, 1)

        else:

            if self.thermodynamic_params_guess is None:

                dh_lower = 10
                dh_upper = 500

            else:

                dh_lower = self.thermodynamic_params_guess[1] / 5
                dh_upper = self.thermodynamic_params_guess[1] * 5

        low_bounds[1] = dh_lower
        high_bounds[1] = dh_upper

        self.cp_value = cp_value
        self.fixed_cp = cp_value is not None

        self.limited_cp = cp_limits is not None and not self.fixed_cp

        if self.limited_cp:

            cp_lower, cp_upper = cp_limits

        else:

            cp_lower, cp_upper = 0.1, 5

        if self.fixed_cp:

            # Remove the Cp from p0, low_bounds and high_bounds
            # Remove Cp0 from the parameter names
            p0 = np.delete(p0, 2)
            low_bounds = np.delete(low_bounds, 2)
            high_bounds = np.delete(high_bounds, 2)
            params_names.pop(2)

        else:

            low_bounds[2] = cp_lower
            high_bounds[2] = cp_upper

            # Verify that the Cp initial guess is within the user-defined limits
            p0[2] = adjust_value_to_interval(p0[2], cp_lower, cp_upper, 0.5)

        id_m = 2 + (not self.fixed_cp)

        low_bounds[id_m] = 0.5
        high_bounds[id_m] = 9

        # Populate the expanded signal and temperature lists
        self.expand_multiple_signal()

        kwargs = {
            'denaturant_concentrations' : self.denaturant_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds' : low_bounds,
            'high_bounds' : high_bounds,
            'cp_value' : cp_value,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'signal_fx' : signal_two_state_tc_unfolding
        }

        fit_fx = fit_tc_unfolding_single_slopes

        # Do a quick prefit with a reduced data set
        if self.pre_fit:

            kwargs['list_of_temperatures'] = self.temp_lst_expanded_subset
            kwargs['list_of_signals'] = self.signal_lst_expanded_subset

            global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

            p0 = global_fit_params

        # Now use the whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        # First fit without m-value dependence on temperature
        global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

        # Insert the initial estimate for the m-value dependence of temperature, in the position 4
        if fit_m_dep:

            kwargs['fit_m1'] = fit_m_dep

            p0 = global_fit_params
            p0 = np.insert(p0, id_m+1, 0)
            low_bounds = np.insert(low_bounds, id_m+1, -0.5)
            high_bounds = np.insert(high_bounds, id_m+1, 0.5)

            kwargs['initial_parameters'] = p0
            kwargs['low_bounds'] = low_bounds
            kwargs['high_bounds'] = high_bounds

            params_names.insert(id_m+1, 'm - T dependence')

            global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds, result, minimizer = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            fit_m_dep,
            self.limited_cp,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
            result=result,
            minimizer=minimizer,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(predicted, self.nr_signals, self.nr_den)

        self.result = result
        self.minimizer = minimizer

        self.global_fit_done = True

        self.fit_m_dep = fit_m_dep

        self.params_names = params_names

        self.create_params_df()
        self.create_dg_df()

        # Add the kwargs and fit_fx to the object for potential later use in leave-one-out analysis
        self.kwargs_fit = kwargs
        self.fit_fx = fit_fx

        if predict_baselines:
            self.predict_baselines()

        return None

    def fit_thermal_unfolding_global_global(self, predict_baselines=True):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters and global slopes (but local baselines)
        Multiple refers to the fact that we fit many signals at the same time, such as 350nm and 330nm
        Must be run after fit_thermal_unfolding_global_multiple

        Notes
        -----
        Updates global fitting attributes and sets `global_global_fit_done` when complete.
        """

        self.global_global_global_fit_done = False  # Reset the flag for the more complex global fit

        # Requires global fit done
        if not self.global_fit_done:
            self.fit_thermal_unfolding_global()

        if self.signal_ids is None:
            self.set_signal_id()

        param_init = 3 + self.fit_m_dep + (self.cp_value is None)

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_den * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]

        low_bounds_p1Ns = self.low_bounds[param_init:param_init + n_datasets]
        low_bounds_p1Us = self.low_bounds[param_init + n_datasets:param_init + 2 * n_datasets]

        high_bounds_p1Ns = self.high_bounds[param_init:param_init + n_datasets]
        high_bounds_p1Us = self.high_bounds[param_init + n_datasets:param_init + 2 * n_datasets]

        id_start = param_init + 2 * n_datasets

        params_names = self.params_names[:id_start]

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p2Ns = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p2Ns = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p2Ns = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p2Us = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p2Us = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p2Us = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p3Ns = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p3Ns = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p3Ns = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p3Us = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p3Us = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p3Us = self.high_bounds[id_start:id_start + n_datasets]

        p0 = np.concatenate([p0, p1Ns, p1Us])
        low_bounds = np.concatenate([low_bounds, low_bounds_p1Ns, low_bounds_p1Us])
        high_bounds = np.concatenate([high_bounds, high_bounds_p1Ns, high_bounds_p1Us])

        # Baselines are still independent for each signal and denaturant concentration
        # Slopes and quadratic terms are shared - per signal only

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            p2Ns = re_arrange_params(p2Ns, self.nr_signals)
            low_bounds_p2Ns = re_arrange_params(low_bounds_p2Ns, self.nr_signals)
            high_bounds_p2Ns = re_arrange_params(high_bounds_p2Ns, self.nr_signals)

            for kNs_i, low_bounds_kNs_i, high_bounds_kNs_i in zip(p2Ns, low_bounds_p2Ns, high_bounds_p2Ns):
                #p0 = np.append(p0, np.median(kNs_i))
                # use a weighted average that gives more weight to the signals with lower denaturant concentrations, as they are more likely to show a clear slope
                p0 = np.append(p0, np.average(kNs_i, weights=1/(np.array(self.denaturant_concentrations)+0.1)))
                low_bounds = np.append(low_bounds, np.min(low_bounds_kNs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_kNs_i))

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            p2Us = re_arrange_params(p2Us, self.nr_signals)
            low_bounds_p2Us = re_arrange_params(low_bounds_p2Us, self.nr_signals)
            high_bounds_p2Us = re_arrange_params(high_bounds_p2Us, self.nr_signals)

            for kUs_i, low_bounds_kUs_i, high_bounds_kUs_i in zip(p2Us, low_bounds_p2Us, high_bounds_p2Us):
                #p0 = np.append(p0, np.median(kUs_i))
                # Use a weighted average that gives more weight to the signals with higher denaturant concentrations, as they are more likely to show a clear slope
                p0 = np.append(p0, np.average(kUs_i, weights=1/(1/(np.array(self.denaturant_concentrations)+0.1))))
                low_bounds = np.append(low_bounds, np.min(low_bounds_kUs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_kUs_i))

        if self.native_baseline_type in ['quadratic', 'exponential']:

            p3Ns = re_arrange_params(p3Ns, self.nr_signals)
            low_bounds_p3Ns = re_arrange_params(low_bounds_p3Ns, self.nr_signals)
            high_bounds_p3Ns = re_arrange_params(high_bounds_p3Ns, self.nr_signals)

            for qNs_i, low_bounds_qNs_i, high_bounds_qNs_i in zip(p3Ns, low_bounds_p3Ns, high_bounds_p3Ns):
                #p0 = np.append(p0, np.median(qNs_i))
                # Use a weighted average that gives more weight to the signals with lower denaturant concentrations, as they are more likely to show a clear curvature
                p0 = np.append(p0, np.average(qNs_i, weights=1/(np.array(self.denaturant_concentrations)+0.1)))
                low_bounds = np.append(low_bounds, np.min(low_bounds_qNs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_qNs_i))

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            p3Us = re_arrange_params(p3Us, self.nr_signals)
            low_bounds_p3Us = re_arrange_params(low_bounds_p3Us, self.nr_signals)
            high_bounds_p3Us = re_arrange_params(high_bounds_p3Us, self.nr_signals)

            for qUs_i, low_bounds_qUs_i, high_bounds_qUs_i in zip(p3Us, low_bounds_p3Us, high_bounds_p3Us):
                #p0 = np.append(p0, np.median(qUs_i))
                # Use a weighted average that gives more weight to the signals with higher denaturant concentrations, as they are more likely to show a clear curvature
                p0 = np.append(p0, np.average(qUs_i, weights=1/(1/(np.array(self.denaturant_concentrations)+0.1))))
                low_bounds = np.append(low_bounds, np.min(low_bounds_qUs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_qUs_i))

        kwargs = {

            'denaturant_concentrations': self.denaturant_concentrations_expanded,
            'list_of_temperatures': self.temp_lst_expanded_subset,
            'list_of_signals': self.signal_lst_expanded_subset,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'cp_value': self.cp_value,
            'fit_m1': self.fit_m_dep,
            'signal_ids':self.signal_ids,
            'baseline_native_fx': self.baseline_N_fx,
            'baseline_unfolded_fx': self.baseline_U_fx,
            'signal_fx' : signal_two_state_tc_unfolding
        }

        fit_fx = fit_tc_unfolding_shared_slopes_many_signals

        if self.pre_fit:
            # Do a pre-fit with a reduced data set
            global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

            p0 = global_fit_params
        # End of pre-fit

        # Use whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds, result, minimizer = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            self.fit_m_dep,
            self.limited_cp,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
            result=result,
            minimizer=minimizer,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_den)

        self.params_names = params_names

        self.result = result
        self.minimizer = minimizer

        self.create_params_df()
        self.create_dg_df()

        self.global_global_fit_done = True

        self.kwargs_fit = kwargs
        self.fit_fx = fit_fx

        if predict_baselines:
            self.predict_baselines()

        return None

    def fit_thermal_unfolding_global_global_global(
            self,
            model_scale_factor=True,
            predict_baselines=True):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters, global slopes and global baselines
        Must be run after fit_thermal_unfolding_global_global

        Parameters
        ----------
        model_scale_factor : bool, optional
            If True, model a scale factor for each denaturant concentration

        Notes
        -----
        Updates many global fitting attributes and sets `global_global_global_fit_done` when complete. If
        `model_scale_factor` is True the method also creates scaled signal attributes:
        - signal_lst_multiple_scaled, predicted_lst_multiple_scaled
        """

        # Requires global global fit done
        if not self.global_global_fit_done:
            self.fit_thermal_unfolding_global_global()

        param_init = 3 + self.fit_m_dep + (self.cp_value is None)

        params_names = self.params_names[:param_init]

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_den * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]

        p1Ns_per_signal = re_arrange_params(p1Ns, self.nr_signals)
        p1Us_per_signal = re_arrange_params(p1Us, self.nr_signals)

        m1s, b1s, m1s_low, b1s_low, m1s_high, b1s_high = [], [], [], [], [], []
        m2s, b2s, m2s_low, b2s_low, m2s_high, b2s_high = [], [], [], [], [], []

        for p1Ns, p1Us in zip(p1Ns_per_signal, p1Us_per_signal):

            # Estimate the slope of bNs versus denaturant concentration
            m1, b1 = fit_line_robust(self.denaturant_concentrations, p1Ns)
            m1_low = m1 / 100 if m1 > 0 else 100 * m1
            m1_high = 100 * m1 if m1 > 0 else m1 / 100
            b1_low = b1 / 100 if b1 > 0 else 100 * b1
            b1_high = 100 * b1 if b1 > 0 else b1 / 100

            # Estimate the slope of bUs versus denaturant concentration
            m2, b2 = fit_line_robust(self.denaturant_concentrations, p1Us)
            m2_low = m2 / 100 if m2 > 0 else 100 * m2
            m2_high = 100 * m2 if m2 > 0 else m2 / 100
            b2_low = b2 / 100 if b2 > 0 else 100 * b2
            b2_high = 100 * b2 if b2 > 0 else b2 / 100

            m1s.append(m1)
            b1s.append(b1)
            m1s_low.append(m1_low)
            b1s_low.append(b1_low)
            m1s_high.append(m1_high)
            b1s_high.append(b1_high)

            m2s.append(m2)
            b2s.append(b2)
            m2s_low.append(m2_low)
            b2s_low.append(b2_low)
            m2s_high.append(m2_high)
            b2s_high.append(b2_high)

        idx = param_init + 2 * n_datasets

        params_names += ['intercept_native - ' + signal_name for signal_name in self.signal_names]
        params_names += ['intercept_unfolded - ' + signal_name for signal_name in self.signal_names]

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            kNs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_kNs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_kNs = self.high_bounds[idx:idx + self.nr_signals]

            idx += self.nr_signals
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            kUs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_kUs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_kUs = self.high_bounds[idx:idx + self.nr_signals]

            idx += self.nr_signals
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        params_names += ['denaturant_slope_term_native - ' + signal_name for signal_name in self.signal_names]
        params_names += ['denaturant_slope_term_unfolded - ' + signal_name for signal_name in self.signal_names]

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            qNs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_qNs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_qNs = self.high_bounds[idx:idx + self.nr_signals]
            idx += self.nr_signals

            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            qUs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_qUs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_qUs = self.high_bounds[idx:idx + self.nr_signals]
            idx += self.nr_signals

            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        p0 = np.concatenate([p0, b1s, b2s])
        low_bounds = np.concatenate([low_bounds, b1s_low, b2s_low])
        high_bounds = np.concatenate([high_bounds, b1s_high, b2s_high])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            p0 = np.concatenate([p0, kNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kNs])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            p0 = np.concatenate([p0, kUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kUs])

        p0 = np.concatenate([p0, m1s, m2s])
        low_bounds = np.concatenate([low_bounds, m1s_low, m2s_low])
        high_bounds = np.concatenate([high_bounds, m1s_high, m2s_high])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            p0 = np.concatenate([p0, qNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qNs])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            p0 = np.concatenate([p0, qUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qUs])

        # Increase the bounds for c_N and c_U
        # Find index in the param names
        for signal_name in self.signal_names:

            c_N_name = 'denaturant_slope_term_native - ' + signal_name
            c_U_name = 'denaturant_slope_term_unfolded - ' + signal_name

            c_N_idx = params_names.index(c_N_name)
            c_U_idx = params_names.index(c_U_name)

            low_bounds[c_N_idx] -= 5
            high_bounds[c_N_idx] += 5

            low_bounds[c_U_idx] -= 5
            high_bounds[c_U_idx] += 5

        # If required, include a scale factor for each denaturant concentration
        if model_scale_factor:
            # The last denaturant concentration is fixed to 1, the rest are fitted
            scale_factors = [1 for _ in range(self.nr_den - 1)]
            scale_factors_low = [0.5882 for _ in range(self.nr_den - 1)]
            scale_factors_high = [1.7 for _ in range(self.nr_den - 1)]

            p0 = np.concatenate([p0, scale_factors])
            low_bounds = np.concatenate([low_bounds, scale_factors_low])
            high_bounds = np.concatenate([high_bounds, scale_factors_high])

            params_names += ['Scale factor - ' + str(d) + ' (M). ID: ' + str(i) for
                             i, d in enumerate(self.denaturant_concentrations)]

            params_names.pop()  # Remove the last one, as it is fixed to 1

        scale_factor_exclude_ids = [self.nr_den - 1] if model_scale_factor else []

        # Do a prefit with a reduced dataset
        kwargs = {

            'list_of_temperatures' : self.temp_lst_expanded_subset,
            'list_of_signals' : self.signal_lst_expanded_subset,
            'signal_ids' : self.signal_ids,
            'denaturant_concentrations': self.denaturant_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'fit_m1': self.fit_m_dep,
            'model_scale_factor':model_scale_factor,
            'cp_value' : self.cp_value,
            'scale_factor_exclude_ids':scale_factor_exclude_ids,
            'signal_fx' : signal_two_state_tc_unfolding,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'fit_native_den_slope' : True,
            'fit_unfolded_den_slope' : True
        }

        fit_fx = fit_tc_unfolding_many_signals

        if self.pre_fit:

            global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

            # Assign the fitted parameters to the initial guess for the full dataset
            p0 = global_fit_params

            # End of prefit with reduced dataset

        # Use the whole dataset
        kwargs['list_of_signals'] = self.signal_lst_expanded
        kwargs['list_of_temperatures'] = self.temp_lst_expanded

        global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

        # Remove scale factors that are not significant
        if model_scale_factor:

            # 3 parameters corresponding to Tm, dH, m
            # plus Cp if fitted
            # plus m1 if fitted
            idx_start = 3 + self.fit_m_dep + (self.cp_value is None)

            native_factor   = 2+np.sum(baseline_fx_name_to_req_params(self.baseline_N_fx))
            unfolded_factor = 2+np.sum(baseline_fx_name_to_req_params(self.baseline_U_fx))

            # Add index according to the native baseline polynomial order
            idx_start += native_factor * self.nr_signals
            # Add index according to the unfolded baseline polynomial order
            idx_start += unfolded_factor * self.nr_signals

            # Take m1 into account, if fitting it
            idx_start += self.fit_m_dep

            for _ in range(5):

                # Sort in ascending order the IDs to exclude
                scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

                n_fixed_factors = len(scale_factor_exclude_ids)
                n_fit_factors = self.nr_den - n_fixed_factors

                if n_fit_factors == 0:
                    break

                sf_params = global_fit_params[idx_start:(idx_start + n_fit_factors)]

                idxs_to_remove = []
                re_fit = False

                # Add dummy variable where we need to skip the index
                for id in scale_factor_exclude_ids:
                    sf_params = np.insert(sf_params, id, np.nan)

                for i, sf in enumerate(sf_params):

                    if i in scale_factor_exclude_ids:
                        continue

                    if 0.9995 <= sf <= 1.0005:
                        # Exclude the scale factor from the fit
                        scale_factor_exclude_ids.append(i)
                        re_fit = True

                        j1 = np.sum(np.array(scale_factor_exclude_ids) < i)
                        j2 = len(idxs_to_remove)

                        idxs_to_remove.append(idx_start + i - j1 + j2)

                if not re_fit:
                    break

                else:

                    for idx in reversed(idxs_to_remove):

                        global_fit_params = np.delete(global_fit_params, idx)
                        low_bounds = np.delete(low_bounds, idx)
                        high_bounds = np.delete(high_bounds, idx)

                        del params_names[idx]

                    kwargs['initial_parameters'] = global_fit_params
                    kwargs['low_bounds'] = low_bounds
                    kwargs['high_bounds'] = high_bounds
                    kwargs['scale_factor_exclude_ids'] = scale_factor_exclude_ids

                    global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

        rel_errors = relative_errors(global_fit_params, cov)
        
        self.params_names = params_names
        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_den)

        self.result = result
        self.minimizer = minimizer

        self.create_params_df()
        self.create_dg_df()

        self.global_global_global_fit_done = True
        
        self.model_scale_factor = model_scale_factor

        # Obtained the scaled signal too
        if model_scale_factor:

            # signal scaled hos one sublist per selected signal type
            signal_scaled    = deepcopy(self.signal_lst_multiple)
            predicted_scaled = deepcopy(self.predicted_lst_multiple)

            for value, param in zip(self.global_fit_params, self.params_names):

                if 'Scale factor' in param:

                    id = int(param.split('(M). ID: ')[-1])

                    for i in range(len(signal_scaled)):
                        signal_scaled[i][id] /= value
                        predicted_scaled[i][id] /= value

            self.signal_lst_multiple_scaled = signal_scaled
            self.predicted_lst_multiple_scaled = predicted_scaled

        # Store the kwargs and fit_fx for potential later use in leave-one-out analysis
        self.kwargs_fit = kwargs
        self.fit_fx = fit_fx

        if predict_baselines:
            self.predict_baselines()

        return None

    def predict_baselines(self):

        baseline_dfs = []

        native_id   = np.argmin(self.denaturant_concentrations)
        unfolded_id = np.argmax(self.denaturant_concentrations)

        native_baseline_params_dict   = find_baseline_params(self.params_df,mode='native')
        unfolded_baseline_params_dict = find_baseline_params(self.params_df,mode='unfolded')

        for i,signal in enumerate(self.signal_names):

            native_params = native_baseline_params_dict.get(signal)
            unfolded_params = unfolded_baseline_params_dict.get(signal)

            # Find the corresponding temperature array
            idx_native   = native_id + i * self.nr_den
            idx_unfolded = unfolded_id + i * self.nr_den  

            temp_native = self.temp_lst_expanded[idx_native]
            temp_unfolded = self.temp_lst_expanded[idx_unfolded]

            temp_native_K = shift_temperature(temp_native)
            temp_unfolded_K = shift_temperature(temp_unfolded)

            if not self.global_global_global_fit_done:

                native_baseline   = self.baseline_N_fx(temp_native_K,0,0, *native_params) # 0 because the denaturant has no effect here
                unfolded_baseline = self.baseline_U_fx(temp_unfolded_K,0,0, *unfolded_params) # 0 because the denaturant has no effect here

            else:

                lowest_den_conc = np.min(self.denaturant_concentrations)
                highest_den_conc = np.max(self.denaturant_concentrations)
                
                native_baseline = self.baseline_N_fx(temp_native_K,lowest_den_conc,*native_params) # 0 because the denaturant has no effect here
                unfolded_baseline = self.baseline_U_fx(temp_unfolded_K,highest_den_conc,*unfolded_params) # 0 because the denaturant has no effect here

            baseline_df = pd.DataFrame({
                'Temperature (°C)': np.concatenate([temp_native, temp_unfolded]),
                'Baseline': np.concatenate([native_baseline, unfolded_baseline]),
                'State': ['Native'] * len(temp_native) + ['Unfolded'] * len(temp_unfolded),
                'Signal': signal
            })

            baseline_dfs.append(baseline_df)

        self.baseline_df = pd.concat(baseline_dfs, ignore_index=True)

        return None

    def leave_one_out_cross_validation(self):

        """
        Perform a leave-one-out cross-validation by fitting the model multiple times, each time leaving out one of the datasets (signal-temperature pairs).
        If we selected two signals, such as 350nm and 330nm, we leave two datasets out at a time.
        """

        fit_fx = self.fit_fx

        Tms = []
        DHs = []
        Cps = []
        m0s = []

        n = len(self.signal_lst_expanded)
        # Correct by the number of signals
        n_corr = int(n / self.nr_signals)

        n_fits = 0

        for i in range(n_corr):

            kwargs = deepcopy(self.kwargs_fit) # We need a deep copy to avoid problems with modifying the original kwargs in place across iterations

            i_to_exclude = [i]

            if n_corr < n:

                # If we have more than one signal, we need to exclude the corresponding datasets for the other signals as well
                for signal_id in range(1, self.nr_signals):
                    i_to_exclude.append(i + signal_id * n_corr)

            loo_temperature_list = [temp for j, temp in enumerate(self.temp_lst_expanded) if j not in i_to_exclude]
            loo_signal_list = [signal for j, signal in enumerate(self.signal_lst_expanded) if j not in i_to_exclude]
            loo_denaturant_concentrations = np.delete(self.denaturant_concentrations_expanded, i_to_exclude)
            
            kwargs['list_of_temperatures']      = loo_temperature_list
            kwargs['list_of_signals']           = loo_signal_list
            kwargs['denaturant_concentrations'] = loo_denaturant_concentrations
            
            # Now we need to adjust the initial parameters, low bounds and high bounds to exclude the parameters corresponding to the left-out dataset(s)

            # Case 1: We have a global fit with local baselines and local slopes
            id_start = 3 + self.fit_m_dep + (self.cp_value is None)

            if not self.global_global_fit_done:
                model = 'global'
            elif self.global_global_fit_done and not self.global_global_global_fit_done:
                model = 'global_global'
            else:                
                model = 'global_global_global'

            params_loo, low_bounds_loo, high_bounds_loo = re_arrange_loo_initial_params(
                model,
                self.native_baseline_type,
                self.unfolded_baseline_type,
                i,
                id_start,
                self.global_fit_params,
                self.low_bounds,
                self.high_bounds,
                self.nr_signals,
                n_corr)

            kwargs['initial_parameters'] = params_loo
            kwargs['low_bounds'] = low_bounds_loo
            kwargs['high_bounds'] = high_bounds_loo

            # Edit the signal IDs for global-global and global-global-global fits, if they are done, to exclude the left-out dataset(s)
            if self.global_global_global_fit_done:

                loo_signal_ids = np.delete(self.signal_ids, i_to_exclude)
                kwargs['signal_ids'] = loo_signal_ids
            
            # Check if we have a global global global fit with scale factor, and in that case we need to adjust the scale_factor_exclude_ids 
            if self.model_scale_factor and model == 'global_global_global':
                
                # We need to create a new list to allow modifications. Read the original list from the kwargs to avoid modifying it in place across iterations
                scale_factor_exclude_ids = [x for x in self.kwargs_fit.get('scale_factor_exclude_ids')] 

                if i in scale_factor_exclude_ids:
                    scale_factor_exclude_ids.remove(i)

                for j, sf in enumerate(scale_factor_exclude_ids):

                    # If the scale factor ID is greater than i, we need to decrease it by 1, because we are removing one dataset before it
                    if sf > i:
                        scale_factor_exclude_ids[j] -= 1

                    # If the scale factor ID is less than i, we don't need to do anything                    

                kwargs['scale_factor_exclude_ids'] = scale_factor_exclude_ids
            
            global_fit_params, _, _, _, _ = fit_fx(**kwargs)

            Tm = global_fit_params[0]

            low_bound_Tm = self.low_bounds[0] # Make sure to compare in Kelvin units
            high_bound_Tm = self.high_bounds[0] # Make sure to compare in Kelvin units

            tm_is_acceptable = Tm >= low_bound_Tm + 0.5 and Tm <= high_bound_Tm - 0.5

            DH = global_fit_params[1]

            DH_is_acceptable = DH >= self.low_bounds[1] + 5 and DH <= self.high_bounds[1] - 5

            if self.cp_value is None:

                Cp = global_fit_params[2]

                Cp_is_acceptable = Cp >= self.low_bounds[2] + 0.1 and Cp <= self.high_bounds[2] - 0.1
   
            id = 2 + (self.cp_value is None)
            m0 = global_fit_params[id]

            m0_is_acceptable = m0 >= self.low_bounds[id] + 0.1 and m0 <= self.high_bounds[id] - 0.1

            keep_fit = tm_is_acceptable and DH_is_acceptable and m0_is_acceptable and (Cp_is_acceptable if self.cp_value is None else True)

            if not keep_fit:
                continue
            else: 
                n_fits += 1

            Tms.append(Tm)
            DHs.append(DH)

            id = 2
            if self.cp_value is None:
                Cps.append(Cp)
                id += 1

            m0s.append(m0) 

        # Create a DataFrame to store the results, with the mean and standard deviation of the parameters across the leave-one-out fits

        params = ['Tm (°C)','ΔHm (kcal / mol)']
        
        if self.cp_value is None:
            params.append('ΔCp (kcal / mol / K)')
        
        params.append('m-value (kcal / mol / M)')

        if n_fits == 0:
            self.loo_df = pd.DataFrame({
                'Parameter': params,
                'LOO_Mean': [np.nan for _ in params],
                'LOO_Std': [np.nan for _ in params],
                'N fits': [0 for _ in params]
            })
            raise ValueError("All leave-one-out fits were rejected based on (thermodynamic) parameter bounds. "
                             "In other words, the fitted parameters (Tm, DH, Cp, m-value) were too close to the specified bounds in all fits."
                             "Please check the bounds, model, and the data quality.")

        loo_values = [np.mean(Tms), np.mean(DHs)]
        if self.cp_value is None:
            loo_values.append(np.mean(Cps))
        loo_values.append(np.mean(m0s))

        loo_std = [np.std(Tms), np.std(DHs)]
        if self.cp_value is None:
            loo_std.append(np.std(Cps))
        loo_std.append(np.std(m0s))

        self.loo_df = pd.DataFrame({
            'Parameter': params,
            'Leave_one_out_mean': loo_values,
            'Leave_one_out_std': loo_std,
            'N fits': n_fits
        })

        return None

    def create_fit_report(self,neff=None):

        """
        Create a fit report using the lmfit result object.
        """

        if self.result is None:
            raise ValueError("No fit result available. Please run a fitting method before creating a fit report.")

        self.fit_report = fit_report(self.result)

        if neff is not None:

            aic, bic = aic_bic_eff(self.result, neff)

            # Find the first line with the string '[[Variables]]'
            # and insert the corrected AIC and BIC values before that line

            report_lines = self.fit_report.split('\n')
            insert_idx = next(i for i, line in enumerate(report_lines) if '[[Variables]]' in line)
            report_lines.insert(insert_idx, f"Corrected AIC (neff={neff}): {aic:.2f}")
            report_lines.insert(insert_idx + 1, f"Corrected BIC (neff={neff}): {bic:.2f}")
            self.fit_report = '\n'.join(report_lines)

        return None

    def calculate_confidence_intervals(self, percentage=0.95):

        # Find the right amount of param names
        desired_params = ['Tm', 'DH', 'Cp', 'm0']
        
        param_names = [x for x in list(self.result.params.keys()) if any(param in x for param in desired_params)]

        ci_results = compute_asymmetric_confidence_intervals(
            minimizer = self.minimizer,
            result = self.result,
            param_names = param_names,
            sigmas = [percentage] 
            # If any of the sigma values is less than 1, that will be interpreted as a probability
            # https://lmfit.github.io/lmfit-py/confidence.html
        )

        self.ci_df = ci_dict_to_summary_df(ci_results,percentage=percentage)

        return None

    def signal_to_df(self, signal_type='raw', scaled=False):
        """
        Create a dataframe with three columns: Temperature, Signal, and Denaturant.
        Optimized for speed by avoiding per-curve DataFrame creation.

        Parameters
        ----------
        signal_type : {'raw', 'fitted', 'derivative'}, optional
            Which signal to include in the dataframe. 'raw' uses experimental data, 'fitted' uses model predictions,
            'derivative' uses the estimated derivative signal.
        scaled : bool, optional
            If True and signal_type == 'fitted' or 'raw', use the scaled versions if available.
        """

        # Flatten all arrays and repeat denaturant values accordingly

        signal_df_list = []

        for i,signal_name in enumerate(self.signal_names):

            if signal_type == 'derivative':

                deriv_lst = self.deriv_lst_multiple[i]
                temp_lst = self.temp_deriv_lst_multiple[i]

                signal_all = np.concatenate(deriv_lst)
                temp_all = np.concatenate(temp_lst)

            else:

                # temperature is shared for the experimental and fitted signals
                temp_lst = self.temp_lst_multiple[i]

                if self.max_points is not None:
                    temp_lst = [subset_data(x, self.max_points) for x in temp_lst]

                temp_all = np.concatenate(temp_lst)

                # fitted data signal does not need subset!
                if signal_type == 'fitted':

                    if not scaled:

                        predicted_lst = self.predicted_lst_multiple[i]

                    else:

                        predicted_lst = self.predicted_lst_multiple_scaled[i]

                    signal_all = np.concatenate(predicted_lst)
                    temp_all = np.concatenate(temp_lst)

                # Signal_type set to 'raw'
                else:

                    if not scaled:

                        signal_lst = self.signal_lst_multiple[i]

                    else:

                        signal_lst = self.signal_lst_multiple_scaled[i]

                    if self.max_points is not None:
                        signal_lst = [subset_data(x, self.max_points) for x in signal_lst]

                    signal_all = np.concatenate(signal_lst)

            denat_all = np.concatenate([
                np.full_like(temp_lst[i], self.denaturant_concentrations[i], dtype=np.float64)
                for i in range(len(temp_lst))
            ])

            # Add an ID column, so we can identify the curves, even with the same denaturant concentration
            id_all = np.concatenate([
                np.full_like(temp_lst[i], i, dtype=np.int32)
                for i in range(len(temp_lst))
            ])

            signal_df = pd.DataFrame({
                'Temperature': temp_all,
                'Signal': signal_all,
                'Denaturant': denat_all,
                'ID': id_all
            })

            signal_df['Label'] = signal_name
            signal_df_list.append(signal_df)

        signal_df = pd.concat(signal_df_list, ignore_index=True)

        return signal_df

    def compare_models(
            self,
            native_baseline_types,
            unfolded_baseline_types,
            global_model_types=['global', 'global_global', 'global_global_global'],
            neff=None,
            gamma=1,
            **kwargs):

        """
        Compare different models with different baseline types and global/local parameters by fitting them and comparing their BIC values.

        Parameters
        ----------
        native_baseline_types : list of str
            List of native baseline types to compare. Each element should be one of 'linear', 'quadratic', 'exponential', or 'constant'.
        unfolded_baseline_types : list of str
            List of unfolded baseline types to compare. Each element should be one of 'linear', 'quadratic', 'exponential', or 'constant'.
        global_model_types : list of str
            List of global model types to fit. Each element should be one of 'global', 'global_global', or 'global_global_global'.
        neff : int, optional
            Effective number of data points to use for AIC, BIC, and EBIC calculation. If None, the total number of data points across all signals and temperatures will be used.
        gamma : float, optional
            Tuning parameter for the Extended BIC (EBIC), typically between 0 and 1 (default: 0.5).
            When gamma=0, EBIC reduces to standard BIC. Higher values impose stronger penalties for model complexity.
        **kwargs
            Additional keyword arguments to pass to fit_thermal_unfolding_global (e.g., fit_m_dep, cp_limits, dh_limits, tm_limits, cp_value).

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing the fitted models and their BIC and EBIC values, sorted by EBIC.
        """

        # We will store the results in a list of dictionaries, and then convert it to a DataFrame at the end
        results = []

        # convert to list if not already a list
        native_baseline_types = transform_to_list(native_baseline_types)
        unfolded_baseline_types = transform_to_list(unfolded_baseline_types)

        for native_baseline_type in native_baseline_types:
            for unfolded_baseline_type in unfolded_baseline_types:

                # Create a copy of the original object to avoid modifying it in place across iterations
                monomer_copy = deepcopy(self)

                # Set the baseline types for the copy
                monomer_copy.native_baseline_type = native_baseline_type
                monomer_copy.unfolded_baseline_type = unfolded_baseline_type

                monomer_copy.estimate_baseline_parameters(
                    native_baseline_type,
                    unfolded_baseline_type,
                    self.window_range_native,
                    self.window_range_unfolded
                )
                
                monomer_copy.fit_thermal_unfolding_global(predict_baselines=False, **kwargs)

                if neff is not None:

                    aic, bic = aic_bic_eff(monomer_copy.result, neff)
                    ebic = extended_bic(monomer_copy.result, neff, gamma=gamma)
                
                else:

                    aic = monomer_copy.result.aic
                    bic = monomer_copy.result.bic
                    # Calculate EBIC with total number of data points
                    n_total = monomer_copy.result.ndata
                    ebic = extended_bic(monomer_copy.result, n_total, gamma=gamma)

                # Store the results in the list if the global option is selected
                if 'global' in global_model_types:
                    results.append({
                        'Native Baseline': native_baseline_type,
                        'Unfolded Baseline': unfolded_baseline_type,
                        'Model Type': 'Local slopes and local intercepts',
                        'Tm': monomer_copy.result.params['Tm'].value,
                        'ΔHm': monomer_copy.result.params['DHm'].value,
                        'ΔCp': monomer_copy.result.params['Cp0'].value if 'Cp0' in monomer_copy.result.params else monomer_copy.cp_value,
                        'm-value': monomer_copy.result.params['m0'].value,
                        'AIC': aic,
                        'BIC': bic,
                        'EBIC': ebic,
                        'Reduced χ²': monomer_copy.result.redchi,
                        'Fit Object': deepcopy(monomer_copy)  # Store the Monomer object for potential later use
                    })

                # If the global-global fit is done, we can also do the global-global fit for the same baseline types
                if 'global_global' in global_model_types or 'global_global_global' in global_model_types:

                    monomer_copy.fit_thermal_unfolding_global_global(predict_baselines=False)

                    if neff is not None:

                        aic, bic = aic_bic_eff(monomer_copy.result, neff)
                        ebic = extended_bic(monomer_copy.result, neff, gamma=gamma)

                    else:
                        
                        aic = monomer_copy.result.aic
                        bic = monomer_copy.result.bic
                        # Calculate EBIC with total number of data points
                        n_total = monomer_copy.result.ndata
                        ebic = extended_bic(monomer_copy.result, n_total, gamma=gamma)

                    # Store the results in the list if the global-global option is selected
                    if 'global_global' in global_model_types:

                        results.append({
                            'Native Baseline': native_baseline_type,
                            'Unfolded Baseline': unfolded_baseline_type,
                            'Model Type': 'Global slopes and local intercepts',
                            'Tm': monomer_copy.result.params['Tm'].value,
                            'ΔHm': monomer_copy.result.params['DHm'].value,
                            'ΔCp': monomer_copy.result.params['Cp0'].value if 'Cp0' in monomer_copy.result.params else monomer_copy.cp_value,
                            'm-value': monomer_copy.result.params['m0'].value,
                            'AIC': aic,
                            'BIC': bic,
                            'EBIC': ebic,
                            'Reduced χ²': monomer_copy.result.redchi,
                            'Fit Object': deepcopy(monomer_copy)  # Store the Monomer object for potential later use
                        })

                    if 'global_global_global' in global_model_types:

                        monomer_copy.fit_thermal_unfolding_global_global_global(predict_baselines=False)

                        if neff is not None:

                            aic, bic = aic_bic_eff(monomer_copy.result, neff)
                            ebic = extended_bic(monomer_copy.result, neff, gamma=gamma)

                        else:

                                aic = monomer_copy.result.aic
                                bic = monomer_copy.result.bic
                                # Calculate EBIC with total number of data points
                                n_total = monomer_copy.result.ndata
                                ebic = extended_bic(monomer_copy.result, n_total, gamma=gamma)

                        results.append({
                            'Native Baseline': native_baseline_type,
                            'Unfolded Baseline': unfolded_baseline_type,
                            'Model Type': 'Global slopes and global intercepts',
                            'Tm': monomer_copy.result.params['Tm'].value,
                            'ΔHm': monomer_copy.result.params['DHm'].value,
                            'ΔCp': monomer_copy.result.params['Cp0'].value if 'Cp0' in monomer_copy.result.params else monomer_copy.cp_value,
                            'm-value': monomer_copy.result.params['m0'].value,
                            'AIC': aic,
                            'BIC': bic,
                            'EBIC': ebic,
                            'Reduced χ²': monomer_copy.result.redchi,
                            'Fit Object': deepcopy(monomer_copy)  # Store the Monomer object for potential later use
                        })

        # Convert the results to a DataFrame and sort by EBIC
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='EBIC').reset_index(drop=True)

        # Remove the Fit Object and save it as an attribute for potential later use
        self.fit_objects = results_df.pop('Fit Object').tolist()

        # Round Tm, ΔH, ΔCp, m-value, and Reduced χ² for better readability
        results_df['Tm'] = temperature_to_celsius(results_df['Tm']).round(1)
        results_df['ΔHm'] = results_df['ΔHm'].round(1)
        results_df['ΔCp'] = results_df['ΔCp'].round(2)
        results_df['m-value'] = results_df['m-value'].round(2)
        results_df['AIC'] = results_df['AIC'].round(2)
        results_df['BIC'] = results_df['BIC'].round(2)
        results_df['EBIC'] = results_df['EBIC'].round(2)
        results_df['Reduced χ²'] = results_df['Reduced χ²'].round(4)

        self.comparison_df = results_df

        return None