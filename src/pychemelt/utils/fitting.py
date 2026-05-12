"""
This module contains helper functions to fit unfolding data
Author: Osvaldo Burastero
"""
import lmfit
import numpy as np
from scipy.optimize     import curve_fit
from scipy.optimize     import least_squares
from numpy.linalg       import pinv

from .math import get_rss, temperature_to_kelvin, temperature_to_celsius

__all__ = [
    "fit_line_robust",
    "fit_quadratic_robust",
    "fit_exponential_robust",
    "fit_thermal_unfolding",
    "fit_tc_unfolding_single_slopes",
    "fit_tc_unfolding_single_slopes_lmfit",
    "fit_tc_unfolding_shared_slopes_many_signals",
    "fit_tc_unfolding_many_signals",
    "fit_oligomer_unfolding_single_slopes",
    "fit_oligomer_unfolding_shared_slopes_many_signals",
    "fit_oligomer_unfolding_many_signals",
    "fit_oligomer_unfolding_three_states_single_slopes",
    "fit_oligomer_unfolding_three_states_shared_slopes_many_signals",
    #"fit_oligomer_unfolding_three_states_many_signals",
]

def baseline_fx_name_to_req_params(baseline_fx_name):
    """
    Determine which baseline parameters are required based on the function name.

    Parameters
    ----------
    baseline_fx_name : str or function object
        baseline type to discern the number of parameters needed

    Returns
    -------
    list
        boolean list of needed parameters for the baseline
    """

    # If baseline_fx_name is not a string, extract the name from the function object
    if not isinstance(baseline_fx_name, str):
        baseline_fx_name = baseline_fx_name.__name__

    if 'constant' in baseline_fx_name:

        return [False, False]

    elif 'linear' in baseline_fx_name:

        return [True, False]

    elif 'quadratic' in baseline_fx_name:

        return [True, True]

    #elif baseline_fx_name == 'exponential':
    else:

        return [True, True]


def fit_line_robust(x,y):

    """
    Fit a line to the data using robust fitting

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    m : float
        Slope of the fitted line
    b : float
        Intercept of the fitted line
    """

    def linear_model(x,params):
        m,b = params
        return m * x + b

    p0 = np.polyfit(x, y, 1)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: linear_model(x, params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1
    )

    m, b = res_robust.x

    return m, b

def fit_quadratic_robust(x,y):

    """
    Fit a quadratic equation to the data using robust fitting

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    a : float
        Quadratic coefficient of the fitted polynomial
    b : float
        Linear coefficient of the fitted polynomial
    c : float
        Constant coefficient of the fitted polynomial
    """

    def model(x,params):
        a,b,c = params
        return a*np.square(x) + b*x + c

    p0 = np.polyfit(x, y, 2)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: model(x, params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1
    )

    a,b,c = res_robust.x

    return a,b,c

def fit_exponential_robust(x,y):

    """
    Fit an exponential function to the data using robust fitting.

    Notes
    -----
    Temperatures should be shifted to the reference (Tref) before calling this function.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    a : float
        Baseline
    c : float
        Pre-exponential factor
    alpha : float
        Exponential factor
    """

    def model(x,a,c,alpha):

        return a + c * np.exp(-alpha * x)

    # Initial parameter estimation by grid search

    rss = np.inf

    alpha_seq = np.logspace(-8, -1, 24)

    p0 = np.array( [np.min(y), np.min(y)/2])
    best_alpha = alpha_seq[0]

    low_bounds = [0, 0]

    high_bounds = [1e7, 1e7]

    for alpha in alpha_seq:

        def fit_fx(x,a,c):

            return a + c * np.exp(-alpha * x)

        params, cov = curve_fit(
            fit_fx,
            x,
            y,
            p0=p0,
            bounds=(low_bounds, high_bounds))

        pred =  fit_fx(x, *params)

        rss_curr = get_rss(y, pred)

        if rss_curr < rss:

            p0 = params
            rss = rss_curr
            best_alpha = alpha

    p0 = p0.tolist() + [best_alpha]

    low_bounds.append(0)
    high_bounds.append(1e6)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: model(x, *params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1,
        bounds=(low_bounds, high_bounds),
    )

    a,c,alpha = res_robust.x

    return a,c,alpha

def fit_thermal_unfolding(
    list_of_temperatures, 
    list_of_signals,
    initial_parameters,
    low_bounds, 
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    Cp,
    list_of_oligomer_conc=None):

    """
    Fit the thermal unfolding profile of many curves at the same time.

    This performs global fitting of shared thermodynamic parameters with per-curve baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Function to calculate the signal based on the parameters

    baseline_native_fx : callable
        function to calculate the native state baseline

    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline

    Cp : float
        Heat capacity change (passed to `signal_fx`)
    list_of_oligomer_conc : list, optional
        List of oligomer concentrations for each dataset (if applicable)

    Returns
    -------
    global_fit_params : numpy.ndarray
        Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters
    predicted_lst : list of numpy.ndarray
        Predicted signals for each dataset based on the fitted parameters
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    # Convert the Tm to kelvin
    initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
    low_bounds[0] = temperature_to_kelvin(low_bounds[0])
    high_bounds[0] = temperature_to_kelvin(high_bounds[0])

    def thermal_unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes or pre-exp terms, folded
            - Single slopes  or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        n_datasets = len(list_of_temperatures)
        Tm, dh     = args[:2]  # Temperature of melting, Enthalpy of unfolding

        intercepts_folded   = args[2:(2 + n_datasets)]
        intercepts_unfolded = args[(2 + n_datasets):(2 + n_datasets * 2)]

        id_param_init = (2 + n_datasets * 2)
        n_params      = n_datasets

        if baseline_native_params[0]:

            p2_Ns = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p2_Ns = np.zeros(n_params)

        if baseline_unfolded_params[0]:

            p2_Us = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p2_Us = np.zeros(n_params)

        if baseline_native_params[1]:

            p3_Ns = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p3_Ns = np.zeros(n_params)

        if baseline_unfolded_params[1]:

            p3_Us = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p3_Us = np.zeros(n_params)

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p1_N = intercepts_folded[i]
            p1_U = intercepts_unfolded[i]

            p2_N = p2_Ns[i]
            p2_U = p2_Us[i]

            p3_N = p3_Ns[i]
            p3_U = p3_Us[i]

            y = signal_fx(
                T, Tm, dh,
                p1_N, p2_N, p3_N,
                p1_U, p2_U, p3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp
            )
            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        thermal_unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
        )

    predicted = thermal_unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    # Convert the Tm to Celsius
    global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst



def fit_tc_unfolding_single_slopes(
    list_of_temperatures,
    list_of_signals,
    denaturant_concentrations,
    initial_parameters,
    low_bounds,
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    list_of_oligomer_conc=None,
    fit_m1=False,
    cp_value=None,
    tm_value=None,
    dh_value=None,
    method='least_squares'):
    """
    Vectorized and optimized version of global thermal unfolding fitting.

        Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    denaturant_concentrations : list
        Denaturant concentrations (one per dataset)
    initial_parameters : array-like
        Initial guess for parameters
    low_bounds : array-like
        Lower bounds for parameters
    high_bounds : array-like
        Upper bounds for parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters
    method : str, optional
        Optimization method ('least_sq' or 'curve_fit')

    Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    """

    # ------------------------------------------------------------
    # Precompute dataset structure
    # ------------------------------------------------------------
    n_datasets = len(list_of_temperatures)
    lengths = np.array([len(T) for T in list_of_temperatures])

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    T_all = np.concatenate(list_of_temperatures)
    y_all = np.concatenate(list_of_signals)

    d_all = np.repeat(denaturant_concentrations, lengths)

    if list_of_oligomer_conc is None:
        c_all = np.zeros_like(T_all, dtype=float)
    else:
        c_all = np.repeat(list_of_oligomer_conc, lengths)

    # ------------------------------------------------------------
    # Baseline parameter requirements
    # ------------------------------------------------------------
    use_p2N, use_p3N = baseline_fx_name_to_req_params(baseline_native_fx)
    use_p2U, use_p3U = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    # Work on copies so the caller's arrays are not modified in place
    initial_parameters = np.array(initial_parameters, dtype=float).copy()
    low_bounds = np.array(low_bounds, dtype=float).copy()
    high_bounds = np.array(high_bounds, dtype=float).copy()

    # Convert Tm-related values to Kelvin
    if tm_value is None:
        initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
        low_bounds[0] = temperature_to_kelvin(low_bounds[0])
        high_bounds[0] = temperature_to_kelvin(high_bounds[0])
    else:
        tm_value = temperature_to_kelvin(tm_value)

    # ------------------------------------------------------------
    # Build lmfit Parameters in the same order as the old vector
    # ------------------------------------------------------------
    params = lmfit.Parameters()
    i = 0

    def add_param(name, value, pmin, pmax, vary=True):
        params.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=vary)

    if tm_value is None:
        add_param("Tm", initial_parameters[i], low_bounds[i], high_bounds[i], vary=True)
        i += 1

    if dh_value is None:
        add_param("DHm", initial_parameters[i], low_bounds[i], high_bounds[i], vary=True)
        i += 1

    if cp_value is None:
        add_param("Cp0", initial_parameters[i], low_bounds[i], high_bounds[i], vary=True)
        i += 1

    add_param("m0", initial_parameters[i], low_bounds[i], high_bounds[i], vary=True)
    i += 1

    if fit_m1:
        add_param("m1", initial_parameters[i], low_bounds[i], high_bounds[i], vary=True)
        i += 1

    for j in range(n_datasets):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j], vary=True)
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j], vary=True)
    i += n_datasets

    if use_p2N:
        for j in range(n_datasets):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j], vary=True)
        i += n_datasets

    if use_p2U:
        for j in range(n_datasets):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j], vary=True)
        i += n_datasets

    if use_p3N:
        for j in range(n_datasets):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j], vary=True)
        i += n_datasets

    if use_p3U:
        for j in range(n_datasets):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j], vary=True)
        i += n_datasets

    # ------------------------------------------------------------
    # Vectorized unfolding model
    # ------------------------------------------------------------
    def unfolding_model(pars):
        idx = 0

        if tm_value is None:
            Tm = pars["Tm"].value
        else:
            Tm = tm_value

        if dh_value is None:
            DHm = pars["DHm"].value
        else:
            DHm = dh_value

        if cp_value is None:
            Cp0 = pars["Cp0"].value
        else:
            Cp0 = cp_value

        m0 = pars["m0"].value

        if fit_m1:
            m1 = pars["m1"].value
        else:
            m1 = 0.0

        p1N = np.array([pars[f"p1N_{j}"].value for j in range(n_datasets)])
        p1U = np.array([pars[f"p1U_{j}"].value for j in range(n_datasets)])

        if use_p2N:
            p2N = np.array([pars[f"p2N_{j}"].value for j in range(n_datasets)])
        else:
            p2N = 0.0

        if use_p2U:
            p2U = np.array([pars[f"p2U_{j}"].value for j in range(n_datasets)])
        else:
            p2U = 0.0

        if use_p3N:
            p3N = np.array([pars[f"p3N_{j}"].value for j in range(n_datasets)])
        else:
            p3N = 0.0

        if use_p3U:
            p3U = np.array([pars[f"p3U_{j}"].value for j in range(n_datasets)])
        else:
            p3U = 0.0

        p1N_all = np.repeat(p1N, lengths)
        p1U_all = np.repeat(p1U, lengths)

        if np.isscalar(p2N):
            p2N_all = p2N
        else:
            p2N_all = np.repeat(p2N, lengths)

        if np.isscalar(p2U):
            p2U_all = p2U
        else:
            p2U_all = np.repeat(p2U, lengths)

        if np.isscalar(p3N):
            p3N_all = p3N
        else:
            p3N_all = np.repeat(p3N, lengths)

        if np.isscalar(p3U):
            p3U_all = p3U
        else:
            p3U_all = np.repeat(p3U, lengths)

        return signal_fx(
            T_all, d_all,
            DHm, Tm, Cp0, m0, m1,
            0, p1N_all, p2N_all, p3N_all,
            0, p1U_all, p2U_all, p3U_all,
            baseline_native_fx,
            baseline_unfolded_fx,
            c_all
        )

    # ------------------------------------------------------------
    # Residual function for lmfit
    # ------------------------------------------------------------
    def residuals(pars):
        return unfolding_model(pars) - y_all

    minimizer = lmfit.Minimizer(residuals, params)
    result = minimizer.minimize(method=method)

    global_fit_params = list(result.params.valuesdict().values())

    # ------------------------------------------------------------
    # Covariance matrix
    # ------------------------------------------------------------
    if result.covar is not None:
        cov = result.covar
    else:
        # Fallback if lmfit could not estimate covariance
        J = result.jac
        dof = len(y_all) - len(result.var_names)
        residual_variance = np.sum(result.residual ** 2) / max(dof, 1)
        cov = pinv(J.T @ J) * residual_variance

    # ------------------------------------------------------------
    # Predict & split per dataset
    # ------------------------------------------------------------
    predicted_all = unfolding_model(result.params)

    predicted_lst = []
    start = 0
    for n in lengths:
        predicted_lst.append(predicted_all[start:start + n])
        start += n

    # Convert Tm back to Celsius for the returned vector
    if tm_value is None:
        global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst, result, minimizer
    
def fit_oligomer_unfolding_single_slopes(
        list_of_temperatures,
        list_of_signals,
        oligomer_concentrations,
        initial_parameters,
        low_bounds,
        high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        cp_value=None,
        tm_value=None,
        dh_value=None,
        method='least_squares',
):
    """
    Vectorized and optimized version of global thermal unfolding fitting. of oligomers

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    oligomer_concentrations : list
        sample concentrations of the oligomeric complex (one per dataset)
    initial_parameters : array-like
        Initial guess for parameters
    low_bounds : array-like
        Lower bounds for parameters
    high_bounds : array-like
        Upper bounds for parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
        lmfit minimization result object
    minimizer : lmfit.minimizer.Minimizer
        lmfit minimizer object
    """

    # ------------------------------------------------------------
    # Precompute dataset structure
    # ------------------------------------------------------------
    n_datasets = len(list_of_temperatures)
    lengths = np.array([len(T) for T in list_of_temperatures])

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    T_all = np.concatenate(list_of_temperatures)
    y_all = np.concatenate(list_of_signals)

    C_all = np.repeat(oligomer_concentrations, lengths)

    # ------------------------------------------------------------
    # Baseline parameter requirements (resolved ONCE)
    # ------------------------------------------------------------
    use_p2N, use_p3N = baseline_fx_name_to_req_params(baseline_native_fx)
    use_p2U, use_p3U = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    # Convert the Tm to kelvin
    if tm_value is None:
        initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
        low_bounds[0] = temperature_to_kelvin(low_bounds[0])
        high_bounds[0] = temperature_to_kelvin(high_bounds[0])
    else:
        tm_value = temperature_to_kelvin(tm_value)

    params_lmfit = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params_lmfit.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    if tm_value is None:
        add_param("Tm", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    if dh_value is None:
        add_param("DHm", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    if cp_value is None:
        add_param("Cp0", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_datasets):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    if use_p2N:
        for j in range(n_datasets):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    if use_p2U:
        for j in range(n_datasets):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    if use_p3N:
        for j in range(n_datasets):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    if use_p3U:
        for j in range(n_datasets):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    def model(pars):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'T_all' containing the temperatures as a single dataset
            - The 'C_all' containing the concentrations as a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Global Cp0
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes or pre-exp terms, folded
            - Single slopes or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """
        if tm_value is None:
            Tm = pars["Tm"].value
        else:
            Tm = tm_value

        if dh_value is None:
            DHm = pars["DHm"].value
        else:
            DHm = dh_value

        if cp_value is None:
            Cp0 = pars["Cp0"].value
        else:
            Cp0 = cp_value

        p1N = np.repeat(np.array([pars[f"p1N_{j}"].value for j in range(n_datasets)]), lengths)
        p1U = np.repeat(np.array([pars[f"p1U_{j}"].value for j in range(n_datasets)]), lengths)

        if use_p2N:
            p2N = np.repeat(np.array([pars[f"p2N_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p2N = 0.0

        if use_p2U:
            p2U = np.repeat(np.array([pars[f"p2U_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p2U = 0.0

        if use_p3N:
            p3N = np.repeat(np.array([pars[f"p3N_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p3N = 0.0

        if use_p3U:
            p3U = np.repeat(np.array([pars[f"p3U_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p3U = 0.0

        # ---- Single vectorized signal evaluation ----
        return signal_fx(
                T_all,C_all, Tm, DHm,
                0, p1N, p2N, p3N,
                0, p1U, p2U, p3U,
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp0,
            )


    def residuals(pars):
        return model(pars) - y_all

    if method == "least_sq":
        lmfit_method = "least_squares"
    elif method == "curve_fit":
        lmfit_method = "least_squares"
    else:
        lmfit_method = method

    minimizer = lmfit.Minimizer(residuals, params_lmfit)
    result = minimizer.minimize(method=lmfit_method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(y_all) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = pinv(J.T @ J) * residual_variance

    predicted_all = model(result.params)

    predicted_lst = []
    start = 0
    for n in lengths:
        predicted_lst.append(predicted_all[start:start + n])
        start += n

    # Convert the Tm back to Celsius
    if tm_value is None:
        global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst, result, minimizer

def fit_oligomer_unfolding_three_states_single_slopes(
        list_of_temperatures,
        list_of_signals,
        oligomer_concentrations,
        initial_parameters,
        low_bounds,
        high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        t1=None,
        t2=None,
        dh1=None,
        dh2=None,
        CpTh_value=None,
        method='least_squares',
):
    """
    Vectorized and optimized version of global thermal unfolding fitting. of oligomers

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    oligomer_concentrations : list
        oligomer concentrations (one per dataset)
    initial_parameters : array-like
        Initial guess for parameters
    low_bounds : array-like
        Lower bounds for parameters
    high_bounds : array-like
        Upper bounds for parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    t1, t2 : float, optional
        Values for the unfolding temperatures one and two
    dh1, dh2 : float, optional
        Values for the unfolding enthalpy one and two
    CpTh_value : float, optional
        Value for the total Cp of the system, enabling fitting of Cp1

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    result : lmfit.minimizer.MinimizerResult
    minimizer : lmfit.minimizer.Minimizer
    """

    # ------------------------------------------------------------
    # Precompute dataset structure
    # ------------------------------------------------------------
    n_datasets = len(list_of_temperatures)
    lengths = np.array([len(T) for T in list_of_temperatures])

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    T_all = np.concatenate(list_of_temperatures)
    y_all = np.concatenate(list_of_signals)

    C_all = np.repeat(oligomer_concentrations, lengths)

    # ------------------------------------------------------------
    # Baseline parameter requirements (resolved ONCE)
    # ------------------------------------------------------------
    use_p2N, use_p3N = baseline_fx_name_to_req_params(baseline_native_fx)
    use_p2U, use_p3U = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    # Convert the Tm to kelvin
    if not t1:
        initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
        low_bounds[0] = temperature_to_kelvin(low_bounds[0])
        high_bounds[0] = temperature_to_kelvin(high_bounds[0])
    else:
        initial_parameters[0]= temperature_to_kelvin(t1)
        low_bounds[0] = initial_parameters[0] - 20
        high_bounds[0] = initial_parameters[0] + 20

    if not t2:
        initial_parameters[2] = temperature_to_kelvin(initial_parameters[2])
        low_bounds[2] = temperature_to_kelvin(low_bounds[2])
        high_bounds[2] = temperature_to_kelvin(high_bounds[2])
    else:
        initial_parameters[2]= temperature_to_kelvin(t2)
        low_bounds[2] = initial_parameters[2] - 20
        high_bounds[2] = initial_parameters[2] + 20

    if dh1:
        initial_parameters[1] = dh1
        low_bounds[1] = dh1 - 50
        high_bounds[1] = dh1 + 50

    if dh2:
        initial_parameters[3] = dh2
        low_bounds[3] = dh2 - 50
        high_bounds[3] = dh2 + 50


    params_lmfit = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params_lmfit.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    add_param("Tm1", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm1", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("Tm2", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm2", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if CpTh_value is not None:
        add_param("Cp1", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_datasets):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"bI_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    if use_p2N:
        for j in range(n_datasets):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    if use_p2U:
        for j in range(n_datasets):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    if use_p3N:
        for j in range(n_datasets):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    if use_p3U:
        for j in range(n_datasets):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_datasets

    def model(pars):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'T_all' containing the temperatures as a single dataset
            - The 'C_all' containing the concentrations as a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature for the first transition
            - Global enthalpy of unfolding for the first transition
            - Global melting temperature for the second transition
            - Global enthalpy of unfolding for the second transition
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single intercepts, intermediate
            - Single slopes or pre-exp terms, folded
            - Single slopes or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        Tm1 = pars["Tm1"].value
        DHm1 = pars["DHm1"].value
        Tm2 = pars["Tm2"].value
        DHm2 = pars["DHm2"].value

        if CpTh_value is not None:
            Cp1 = pars["Cp1"].value
            CpTh = CpTh_value
        else:
            Cp1 = 0.0
            CpTh = 0.0

        p1N = np.repeat(np.array([pars[f"p1N_{j}"].value for j in range(n_datasets)]), lengths)
        p1U = np.repeat(np.array([pars[f"p1U_{j}"].value for j in range(n_datasets)]), lengths)
        bI = np.repeat(np.array([pars[f"bI_{j}"].value for j in range(n_datasets)]), lengths)

        if use_p2N:
            p2N = np.repeat(np.array([pars[f"p2N_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p2N = 0.0

        if use_p2U:
            p2U = np.repeat(np.array([pars[f"p2U_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p2U = 0.0

        if use_p3N:
            p3N = np.repeat(np.array([pars[f"p3N_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p3N = 0.0

        if use_p3U:
            p3U = np.repeat(np.array([pars[f"p3U_{j}"].value for j in range(n_datasets)]), lengths)
        else:
            p3U = 0.0

        # ---- Single vectorized signal evaluation ----
        return signal_fx(
                T_all,C_all, Tm1, DHm1, Tm2, DHm2,
                0, p1N, p2N, p3N,
                0, p1U, p2U, p3U,
                baseline_native_fx,
                baseline_unfolded_fx,
                bI,
                Cp1,CpTh,
            )


    def residuals(pars):
        return model(pars) - y_all

    if method == "least_sq":
        lmfit_method = "least_squares"
    elif method == "curve_fit":
        lmfit_method = "least_squares"
    else:
        lmfit_method = method

    minimizer = lmfit.Minimizer(residuals, params_lmfit)
    result = minimizer.minimize(method=lmfit_method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(y_all) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = pinv(J.T @ J) * residual_variance

    predicted_all = model(result.params)

    predicted_lst = []
    start = 0
    for n in lengths:
        predicted_lst.append(predicted_all[start:start + n])
        start += n

    # Convert the Tm back to Celsius

    global_fit_params[0] = temperature_to_celsius(global_fit_params[0])
    global_fit_params[2] = temperature_to_celsius(global_fit_params[2])

    return global_fit_params, cov, predicted_lst, result, minimizer


def fit_tc_unfolding_shared_slopes_many_signals(
    list_of_temperatures,
    list_of_signals,
    signal_ids,
    denaturant_concentrations,
    initial_parameters,
    low_bounds,
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    list_of_oligomer_conc=None,
    fit_m1=False,
    cp_value=None,
    tm_value=None,
    dh_value=None,
    method='least_squares'
):
    """
    Vectorized fitting of thermochemical unfolding curves for multiple signal types
    sharing thermodynamic parameters and slopes, using lmfit.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
     baseline_native_fx : callable
        function to calculate the baseline for the native state
    baseline_unfolded_fx : callable
        function to calculate the baseline for the unfolded state
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters
    method : str, optional
        Optimization method for lmfit minimizer. Defaults to 'least_squares'.

    Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
        lmfit minimization result object
    minimizer : lmfit.minimizer.Minimizer
        lmfit minimizer object
    """

    # Flatten all signals
    all_signal = np.concatenate(list_of_signals, axis=0)
    n_signals = np.max(signal_ids) + 1
    n_datasets = len(list_of_temperatures)

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    # Precompute indices for slicing the flattened concatenated arrays
    dataset_starts = np.cumsum([0] + [len(T) for T in list_of_temperatures][:-1])
    dataset_ends = np.cumsum([len(T) for T in list_of_temperatures])

    # Convert the Tm to kelvin
    if tm_value is None:
        initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
        low_bounds[0] = temperature_to_kelvin(low_bounds[0])
        high_bounds[0] = temperature_to_kelvin(high_bounds[0])
    else:
        tm_value = temperature_to_kelvin(tm_value)

    params = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    if tm_value is None:
        add_param("Tm", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    if dh_value is None:
        add_param("DHm", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    if cp_value is None:
        add_param("Cp0", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    add_param("m0", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if fit_m1:
        add_param("m1", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_datasets):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    if baseline_native_params[0]:
        for j in range(n_signals):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[0]:
        for j in range(n_signals):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[1]:
        for j in range(n_signals):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[1]:
        for j in range(n_signals):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    def residuals(pars):
        if tm_value is None:
            Tm = pars["Tm"].value
        else:
            Tm = tm_value

        if dh_value is None:
            DHm = pars["DHm"].value
        else:
            DHm = dh_value

        if cp_value is None:
            Cp0 = pars["Cp0"].value
        else:
            Cp0 = cp_value

        m0 = pars["m0"].value
        m1 = pars["m1"].value if fit_m1 else 0

        intercepts_folded = np.array([pars[f"p1N_{j}"].value for j in range(n_datasets)])
        intercepts_unfolded = np.array([pars[f"p1U_{j}"].value for j in range(n_datasets)])

        # Shared slopes / coefficients per signal type
        if baseline_native_params[0]:
            p2_n_s = np.array([pars[f"p2N_{j}"].value for j in range(n_signals)])
        else:
            p2_n_s = np.zeros(n_signals)

        if baseline_unfolded_params[0]:
            p2_u_s = np.array([pars[f"p2U_{j}"].value for j in range(n_signals)])
        else:
            p2_u_s = np.zeros(n_signals)

        if baseline_native_params[1]:
            p3_n_s = np.array([pars[f"p3N_{j}"].value for j in range(n_signals)])
        else:
            p3_n_s = np.zeros(n_signals)

        if baseline_unfolded_params[1]:
            p3_u_s = np.array([pars[f"p3U_{j}"].value for j in range(n_signals)])
        else:
            p3_u_s = np.zeros(n_signals)

        # Vectorized evaluation for all datasets
        predicted_all = np.zeros_like(all_signal)
        for i, T in enumerate(list_of_temperatures):
            start, end = dataset_starts[i], dataset_ends[i]
            d = denaturant_concentrations[i]
            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]
            sig_id = signal_ids[i]

            predicted_all[start:end] = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                0, intercepts_folded[i], p2_n_s[sig_id], p3_n_s[sig_id],
                0, intercepts_unfolded[i], p2_u_s[sig_id], p3_u_s[sig_id],
                baseline_native_fx,
                baseline_unfolded_fx,
                c
            )

        return predicted_all - all_signal

    minimizer = lmfit.Minimizer(residuals, params)
    result = minimizer.minimize(method=method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(all_signal) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = np.linalg.pinv(J.T @ J) * residual_variance

    # Convert predicted signal into list of arrays per dataset
    predicted = all_signal + result.residual
    predicted_lst = [predicted[start:end] for start, end in zip(dataset_starts, dataset_ends)]

    # Convert the Tm back to Celsius
    if tm_value is None:
        global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst, result, minimizer

def fit_oligomer_unfolding_shared_slopes_many_signals(
    list_of_temperatures,
    list_of_signals,
    signal_ids,
    oligomer_concentrations,
    initial_parameters,
    low_bounds,
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    cp_value=None,
    tm_value=None,
    dh_value=None,
    method='least_squares'
):
    """
    Vectorized fitting of oligomer thermal unfolding curves for multiple signal types
    sharing thermodynamic parameters and slopes, using lmfit.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset.
    list_of_signals : list of array-like
        Signal arrays for each dataset.
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    oligomer_concentrations : list
        sample concentrations of the oligomeric complex for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
     baseline_native_fx : callable
        function to calculate the baseline for the native state
    baseline_unfolded_fx : callable
        function to calculate the baseline for the unfolded state
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
        lmfit minimization result object
    minimizer : lmfit.minimizer.Minimizer
        lmfit minimizer object

    """

    # Flatten all signals
    all_signal = np.concatenate(list_of_signals, axis=0)
    n_signals = np.max(signal_ids) + 1
    n_datasets = len(list_of_temperatures)

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    # Precompute indices for slicing the flattened concatenated arrays
    dataset_starts = np.cumsum([0] + [len(T) for T in list_of_temperatures][:-1])
    dataset_ends = np.cumsum([len(T) for T in list_of_temperatures])

    # Convert the Tm to kelvin
    if tm_value is None:
        initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
        low_bounds[0] = temperature_to_kelvin(low_bounds[0])
        high_bounds[0] = temperature_to_kelvin(high_bounds[0])
    else:
        tm_value = temperature_to_kelvin(tm_value)

    params_lmfit = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params_lmfit.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    if tm_value is None:
        add_param("Tm", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    if dh_value is None:
        add_param("DHm", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    if cp_value is None:
        add_param("Cp0", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_datasets):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    if baseline_native_params[0]:
        for j in range(n_signals):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[0]:
        for j in range(n_signals):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[1]:
        for j in range(n_signals):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[1]:
        for j in range(n_signals):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    def residuals(pars):
        if tm_value is None:
            Tm = pars["Tm"].value
        else:
            Tm = tm_value

        if dh_value is None:
            DHm = pars["DHm"].value
        else:
            DHm = dh_value

        if cp_value is None:
            Cp0 = pars["Cp0"].value
        else:
            Cp0 = cp_value

        intercepts_folded = np.array([pars[f"p1N_{j}"].value for j in range(n_datasets)])
        intercepts_unfolded = np.array([pars[f"p1U_{j}"].value for j in range(n_datasets)])

        if baseline_native_params[0]:
            p2_n_s = np.array([pars[f"p2N_{j}"].value for j in range(n_signals)])
        else:
            p2_n_s = np.zeros(n_signals)

        if baseline_unfolded_params[0]:
            p2_u_s = np.array([pars[f"p2U_{j}"].value for j in range(n_signals)])
        else:
            p2_u_s = np.zeros(n_signals)

        if baseline_native_params[1]:
            p3_n_s = np.array([pars[f"p3N_{j}"].value for j in range(n_signals)])
        else:
            p3_n_s = np.zeros(n_signals)

        if baseline_unfolded_params[1]:
            p3_u_s = np.array([pars[f"p3U_{j}"].value for j in range(n_signals)])
        else:
            p3_u_s = np.zeros(n_signals)

        # Vectorized evaluation for all datasets
        predicted_all = np.zeros_like(all_signal)
        for i, T in enumerate(list_of_temperatures):
            start, end = dataset_starts[i], dataset_ends[i]
            c = oligomer_concentrations[i]
            sig_id = signal_ids[i]

            predicted_all[start:end] = signal_fx(
                T, c, Tm, DHm,
                0, intercepts_folded[i], p2_n_s[sig_id], p3_n_s[sig_id],
                0, intercepts_unfolded[i], p2_u_s[sig_id], p3_u_s[sig_id],
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp0
            )


        return predicted_all - all_signal

    if method == "least_sq":
        lmfit_method = "least_squares"
    elif method == "curve_fit":
        lmfit_method = "least_squares"
    else:
        lmfit_method = method

    minimizer = lmfit.Minimizer(residuals, params_lmfit)
    result = minimizer.minimize(method=lmfit_method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(all_signal) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = np.linalg.pinv(J.T @ J) * residual_variance

    # Convert predicted signal into list of arrays per dataset
    predicted = all_signal + result.residual
    predicted_lst = [predicted[start:end] for start, end in zip(dataset_starts, dataset_ends)]

    # Convert the Tm back to Celsius
    if tm_value is None:
        global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst, result, minimizer

def fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
    list_of_temperatures,
    list_of_signals,
    signal_ids,
    oligomer_concentrations,
    initial_parameters,
    low_bounds,
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    t1=None,
    t2=None,
    dh1=None,
    dh2=None,
    CpTh_value=None,
    method='least_squares',
):
    """
    Vectorized fitting of oligomer thermal unfolding curves for multiple signal types
    sharing thermodynamic parameters and slopes, using lmfit.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset.
    list_of_signals : list of array-like
        Signal arrays for each dataset.
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    oligomer_concentrations : list
        Oligomer concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
     baseline_native_fx : callable
        function to calculate the baseline for the native state
    baseline_unfolded_fx : callable
        function to calculate the baseline for the unfolded state
    t1, t2 : float, optional
        Values for the unfolding temperatures one and two
    dh1, dh2 : float, optional
        Values for the unfolding enthalpy one and two
    CpTh_value : float, optional
        Value for the total Cp of the system, enabling fitting of Cp1

    Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
    minimizer : lmfit.minimizer.Minimizer

    """

    # Flatten all signals
    all_signal = np.concatenate(list_of_signals, axis=0)
    n_signals = np.max(signal_ids) + 1
    n_datasets = len(list_of_temperatures)

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    # Precompute indices for slicing the flattened concatenated arrays
    dataset_starts = np.cumsum([0] + [len(T) for T in list_of_temperatures][:-1])
    dataset_ends = np.cumsum([len(T) for T in list_of_temperatures])

    # Convert the Tm to kelvin
    if not t1:
        initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
        low_bounds[0] = temperature_to_kelvin(low_bounds[0])
        high_bounds[0] = temperature_to_kelvin(high_bounds[0])
    else:
        initial_parameters[0]= temperature_to_kelvin(t1)
        low_bounds[0] = np.max([initial_parameters[0] - 20, 271])
        high_bounds[0] = initial_parameters[0] + 20

    if not t2:
        initial_parameters[2] = temperature_to_kelvin(initial_parameters[2])
        low_bounds[2] = temperature_to_kelvin(low_bounds[2])
        high_bounds[2] = temperature_to_kelvin(high_bounds[2])
    else:
        initial_parameters[2]= temperature_to_kelvin(t2)
        low_bounds[2] = np.max([initial_parameters[2] - 20, 271])
        high_bounds[2] = initial_parameters[2] + 20

    if dh1:
        initial_parameters[1] = dh1
        low_bounds[1] = dh1 - 50
        high_bounds[1] = dh1 + 50

    if dh2:
        initial_parameters[3] = dh2
        low_bounds[3] = dh2 - 50
        high_bounds[3] = dh2 + 50

    params_lmfit = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params_lmfit.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    add_param("Tm1", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm1", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("Tm2", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm2", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if CpTh_value is not None:
        add_param("Cp1", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_datasets):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    for j in range(n_datasets):
        add_param(f"bI_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_datasets

    if baseline_native_params[0]:
        for j in range(n_signals):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[0]:
        for j in range(n_signals):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[1]:
        for j in range(n_signals):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[1]:
        for j in range(n_signals):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    def residuals(pars):
        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature for the first transition
            - Global enthalpy of unfolding for the first transition
            - Global melting temperature for the second transition
            - Global enthalpy of unfolding for the second transition
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single intercepts, intermediate
            - Single slopes or pre-exp terms, folded
            - Single slopes or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        Tm1 = pars["Tm1"].value
        DHm1 = pars["DHm1"].value
        Tm2 = pars["Tm2"].value
        DHm2 = pars["DHm2"].value

        if CpTh_value is not None:
            Cp1 = pars["Cp1"].value
            CpTh = CpTh_value
        else:
            Cp1 = 0.0
            CpTh = 0.0

        intercepts_folded = np.array([pars[f"p1N_{j}"].value for j in range(n_datasets)])
        intercepts_unfolded = np.array([pars[f"p1U_{j}"].value for j in range(n_datasets)])
        intercepts_intermediates = np.array([pars[f"bI_{j}"].value for j in range(n_datasets)])

        if baseline_native_params[0]:
            p2_n_s = np.array([pars[f"p2N_{j}"].value for j in range(n_signals)])
        else:
            p2_n_s = np.zeros(n_signals)

        if baseline_unfolded_params[0]:
            p2_u_s = np.array([pars[f"p2U_{j}"].value for j in range(n_signals)])
        else:
            p2_u_s = np.zeros(n_signals)

        if baseline_native_params[1]:
            p3_n_s = np.array([pars[f"p3N_{j}"].value for j in range(n_signals)])
        else:
            p3_n_s = np.zeros(n_signals)

        if baseline_unfolded_params[1]:
            p3_u_s = np.array([pars[f"p3U_{j}"].value for j in range(n_signals)])
        else:
            p3_u_s = np.zeros(n_signals)

        # Vectorized evaluation for all datasets
        predicted_all = np.zeros_like(all_signal)
        for i, T in enumerate(list_of_temperatures):
            start, end = dataset_starts[i], dataset_ends[i]
            c = oligomer_concentrations[i]
            sig_id = signal_ids[i]

            predicted_all[start:end] = signal_fx(
                T, c, Tm1, DHm1, Tm2, DHm2,
                0, intercepts_folded[i], p2_n_s[sig_id], p3_n_s[sig_id],
                0, intercepts_unfolded[i], p2_u_s[sig_id], p3_u_s[sig_id],
                baseline_native_fx,
                baseline_unfolded_fx,
                intercepts_intermediates[i],
                Cp1, CpTh,
            )

        return predicted_all - all_signal

    if method == "least_sq":
        lmfit_method = "least_squares"
    elif method == "curve_fit":
        lmfit_method = "least_squares"
    else:
        lmfit_method = method

    minimizer = lmfit.Minimizer(residuals, params_lmfit)
    result = minimizer.minimize(method=lmfit_method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(all_signal) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = np.linalg.pinv(J.T @ J) * residual_variance

    # Convert predicted signal into list of arrays per dataset
    predicted = all_signal + result.residual
    predicted_lst = [predicted[start:end] for start, end in zip(dataset_starts, dataset_ends)]

    # Convert the Tm back to Celsius
    global_fit_params[0] = temperature_to_celsius(global_fit_params[0])
    global_fit_params[2] = temperature_to_celsius(global_fit_params[2])

    return global_fit_params, cov, predicted_lst, result, minimizer

def fit_tc_unfolding_many_signals(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        oligomer_concentrations=None,
        fit_m1=False,
        model_scale_factor=False,
        scale_factor_exclude_ids=[],
        cp_value=None,
        method='least_squares',
        fit_native_den_slope=True,
        fit_unfolded_den_slope=True):
    """
    Fit thermochemical unfolding curves for many signals using lmfit.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset.
    list_of_signals : list of array-like
        Signal arrays for each dataset.
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    oligomer_concentrations : list, optional
        Oligomer concentrations per dataset (used by oligomeric models)
    fit_m1 : bool, optional
        Whether to include and fit temperature dependence of the m-value (m1)
    model_scale_factor : bool, optional
        If True, include a per-denaturant concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1
    cp_value : float or None, optional
        If provided, Cp is fixed to this value and not fitted
    method : str, optional
        Optimization method for lmfit minimizer. Defaults to 'least_squares'.
    fit_native_den_slope, fit_unfolded_den_slope : bool, optional
        Whether to fit denaturant dependence of baselines.

    Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
        lmfit minimization result object
    minimizer : lmfit.minimizer.Minimizer
        lmfit minimizer object
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    nr_den = int(len(denaturant_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    baseline_native_params = [fit_native_den_slope] + baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = [fit_unfolded_den_slope] + baseline_fx_name_to_req_params(baseline_unfolded_fx)

    initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
    low_bounds[0] = temperature_to_kelvin(low_bounds[0])
    high_bounds[0] = temperature_to_kelvin(high_bounds[0])

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    params = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    add_param("Tm", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if cp_value is None:
        add_param("Cp0", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    add_param("m0", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if fit_m1:
        add_param("m1", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_signals):
        add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    for j in range(n_signals):
        add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    if baseline_native_params[1]:
        for j in range(n_signals):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[1]:
        for j in range(n_signals):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[0]:
        for j in range(n_signals):
            add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[0]:
        for j in range(n_signals):
            add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[2]:
        for j in range(n_signals):
            add_param(f"p4N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[2]:
        for j in range(n_signals):
            add_param(f"p4U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    sf_fit_ids = []
    if model_scale_factor:
        sf_fit_ids = [k for k in range(nr_den) if k not in scale_factor_exclude_ids]
        for k in sf_fit_ids:
            add_param(f"sf_{k}", initial_parameters[i], low_bounds[i], high_bounds[i])
            i += 1

    def model(pars):
        Tm = pars["Tm"].value
        DHm = pars["DHm"].value
        Cp0 = pars["Cp0"].value if cp_value is None else cp_value
        m0 = pars["m0"].value
        m1 = pars["m1"].value if fit_m1 else 0

        p2_Ns = np.array([pars[f"p2N_{j}"].value for j in range(n_signals)])
        p2_Us = np.array([pars[f"p2U_{j}"].value for j in range(n_signals)])

        if baseline_native_params[1]:
            p3_Ns = np.array([pars[f"p3N_{j}"].value for j in range(n_signals)])
        else:
            p3_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[1]:
            p3_Us = np.array([pars[f"p3U_{j}"].value for j in range(n_signals)])
        else:
            p3_Us = np.zeros(n_signals)

        if baseline_native_params[0]:
            p1_Ns = np.array([pars[f"p1N_{j}"].value for j in range(n_signals)])
        else:
            p1_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[0]:
            p1_Us = np.array([pars[f"p1U_{j}"].value for j in range(n_signals)])
        else:
            p1_Us = np.zeros(n_signals)

        if baseline_native_params[2]:
            p4_Ns = np.array([pars[f"p4N_{j}"].value for j in range(n_signals)])
        else:
            p4_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[2]:
            p4_Us = np.array([pars[f"p4U_{j}"].value for j in range(n_signals)])
        else:
            p4_Us = np.zeros(n_signals)

        if model_scale_factor:
            sf = np.ones(nr_den)
            for k in sf_fit_ids:
                sf[k] = pars[f"sf_{k}"].value
            factors = np.tile(sf, n_signals)
        else:
            factors = None

        signal = []
        for idx, T in enumerate(list_of_temperatures):
            p1_N = p1_Ns[signal_ids[idx]]
            p1_U = p1_Us[signal_ids[idx]]
            p2_N = p2_Ns[signal_ids[idx]]
            p2_U = p2_Us[signal_ids[idx]]
            p3_N = p3_Ns[signal_ids[idx]]
            p3_U = p3_Us[signal_ids[idx]]
            p4_N = p4_Ns[signal_ids[idx]]
            p4_U = p4_Us[signal_ids[idx]]

            d = denaturant_concentrations[idx]
            c = 0 if oligomer_concentrations is None else oligomer_concentrations[idx]

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                p1_N, p2_N, p3_N, p4_N,
                p1_U, p2_U, p3_U, p4_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                c
            )

            scale_factor = 1 if factors is None else factors[idx]
            signal.append(y * scale_factor)

        return np.concatenate(signal, axis=0)

    def residuals(pars):
        return model(pars) - all_signal

    minimizer = lmfit.Minimizer(residuals, params)
    result = minimizer.minimize(method=method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(all_signal) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = np.linalg.pinv(J.T @ J) * residual_variance

    predicted = model(result.params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    # Convert the Tm to Celsius
    global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst, result, minimizer

def fit_oligomer_unfolding_many_signals(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        oligomer_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        model_scale_factor=False,
        scale_factor_exclude_ids=[],
        cp_value=None,
        method='least_squares'):
    """
    Fit thermal unfolding curves of oligomers for many signals (optimized variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    oligomer_concentrations : list
        sample concentrations of the oligomeric complex for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    model_scale_factor : bool, optional
        If True, include a per-oligomeric concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1
    cp_value : float or None, optional
        If provided, Cp is fixed to this value and not fitted

   Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
        lmfit minimization result object
    minimizer : lmfit.minimizer.Minimizer
        lmfit minimizer object
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    nr_olig = int(len(oligomer_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
    low_bounds[0] = temperature_to_kelvin(low_bounds[0])
    high_bounds[0] = temperature_to_kelvin(high_bounds[0])

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    params_lmfit = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params_lmfit.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    add_param("Tm", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if cp_value is None:
        add_param("Cp0", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_signals):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    for j in range(n_signals):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    if baseline_native_params[0]:
        for j in range(n_signals):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[0]:
        for j in range(n_signals):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[1]:
        for j in range(n_signals):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[1]:
        for j in range(n_signals):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if model_scale_factor:
        n_fit_factors = nr_olig - len(scale_factor_exclude_ids)
        for j in range(n_fit_factors):
            add_param(f"sf_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_fit_factors

    def model(pars):

        """
        The parameters order is as follows:

            Tm, Dh, Cp0

            Intercept folded
            Intercept unfolded

            Temperature slope or term pre-exponential factor folded
            Temperature slope term or pre-exponential factor unfolded

            Quadratic coefficient or exponential coefficient folded
            Quadratic coefficient or exponential coefficient unfolded

        """

        Tm = pars["Tm"].value
        DHm = pars["DHm"].value
        if cp_value is None:
            Cp0 = pars["Cp0"].value
        else:
            Cp0 = cp_value

        p1_Ns = np.array([pars[f"p1N_{j}"].value for j in range(n_signals)])
        p1_Us = np.array([pars[f"p1U_{j}"].value for j in range(n_signals)])

        if baseline_native_params[0]:
            p2_Ns = np.array([pars[f"p2N_{j}"].value for j in range(n_signals)])
        else:
            p2_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[0]:
            p2_Us = np.array([pars[f"p2U_{j}"].value for j in range(n_signals)])
        else:
            p2_Us = np.zeros(n_signals)

        if baseline_native_params[1]:
            p3_Ns = np.array([pars[f"p3N_{j}"].value for j in range(n_signals)])
        else:
            p3_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[1]:
            p3_Us = np.array([pars[f"p3U_{j}"].value for j in range(n_signals)])
        else:
            p3_Us = np.zeros(n_signals)

        if model_scale_factor:
            n_fit_factors = nr_olig - len(scale_factor_exclude_ids)
            factors = np.array([pars[f"sf_{j}"].value for j in range(n_fit_factors)])
            for id_ex in scale_factor_exclude_ids:
                factors = np.insert(factors, id_ex, 1.0)
            factors = np.tile(factors, n_signals)

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p1_N = p1_Ns[signal_ids[i]]
            p1_U = p1_Us[signal_ids[i]]
            p2_N = p2_Ns[signal_ids[i]]
            p2_U = p2_Us[signal_ids[i]]
            p3_N = p3_Ns[signal_ids[i]]
            p3_U = p3_Us[signal_ids[i]]

            c = oligomer_concentrations[i]


            y = signal_fx(
                T, c, Tm, DHm,
                0, p1_N, p2_N, p3_N,
                0, p1_U, p2_U, p3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp0
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        return np.concatenate(signal, axis=0)

    def residuals(pars):
        return model(pars) - all_signal

    if method == "least_sq":
        lmfit_method = "least_squares"
    elif method == "curve_fit":
        lmfit_method = "least_squares"
    else:
        lmfit_method = method

    minimizer = lmfit.Minimizer(residuals, params_lmfit)
    result = minimizer.minimize(method=lmfit_method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(all_signal) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = pinv(J.T @ J) * residual_variance

    predicted = model(result.params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    # Convert the Tm to Celsius
    global_fit_params[0] = temperature_to_celsius(global_fit_params[0])

    return global_fit_params, cov, predicted_lst, result, minimizer

def fit_oligomer_unfolding_three_states_many_signals(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        oligomer_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        CpTh_value=None,
        model_scale_factor=False,
        scale_factor_exclude_ids=[],
        method='least_squares'):
    """
    Fit thermal unfolding curves of oligomers for many signals (optimized variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    oligomer_concentrations : list
        oligomer concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    CpTh_value : float, optional
        Value for the total Cp of the system, enabling fitting of Cp1
    model_scale_factor : bool, optional
        If True, include a per-oligomeric concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1
   Returns
    -------
    global_fit_params : numpy.ndarray
         Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix
    predicted_lst : list of numpy.ndarray
        Predicted signals per dataset
    result : lmfit.minimizer.MinimizerResult
    minimizer : lmfit.minimizer.Minimizer
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    nr_olig = int(len(oligomer_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    initial_parameters[0] = temperature_to_kelvin(initial_parameters[0])
    low_bounds[0] = temperature_to_kelvin(low_bounds[0])
    high_bounds[0] = temperature_to_kelvin(high_bounds[0])

    initial_parameters[2] = temperature_to_kelvin(initial_parameters[2])
    low_bounds[2] = temperature_to_kelvin(low_bounds[2])
    high_bounds[2] = temperature_to_kelvin(high_bounds[2])

    list_of_temperatures = [temperature_to_kelvin(T) for T in list_of_temperatures]

    params_lmfit = lmfit.Parameters()
    param_names = []
    i = 0

    def add_param(name, value, pmin, pmax):
        params_lmfit.add(name, value=float(value), min=float(pmin), max=float(pmax), vary=True)
        param_names.append(name)

    add_param("Tm1", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm1", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("Tm2", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1
    add_param("DHm2", initial_parameters[i], low_bounds[i], high_bounds[i])
    i += 1

    if CpTh_value is not None:
        add_param("Cp1", initial_parameters[i], low_bounds[i], high_bounds[i])
        i += 1

    for j in range(n_signals):
        add_param(f"p1N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    for j in range(n_signals):
        add_param(f"p1U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    for j in range(n_signals):
        add_param(f"bI_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
    i += n_signals

    if baseline_native_params[0]:
        for j in range(n_signals):
            add_param(f"p2N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[0]:
        for j in range(n_signals):
            add_param(f"p2U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_native_params[1]:
        for j in range(n_signals):
            add_param(f"p3N_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if baseline_unfolded_params[1]:
        for j in range(n_signals):
            add_param(f"p3U_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_signals

    if model_scale_factor:
        n_fit_factors = nr_olig - len(scale_factor_exclude_ids)
        for j in range(n_fit_factors):
            add_param(f"sf_{j}", initial_parameters[i + j], low_bounds[i + j], high_bounds[i + j])
        i += n_fit_factors

    def model(pars):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature for the first transition
            - Global enthalpy of unfolding for the first transition
            - Global melting temperature for the second transition
            - Global enthalpy of unfolding for the second transition
            - Optional global Cp1 value
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single intercepts, intermediate
            - Single slopes or pre-exp terms, folded
            - Single slopes or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """
        Tm1 = pars["Tm1"].value
        DHm1 = pars["DHm1"].value
        Tm2 = pars["Tm2"].value
        DHm2 = pars["DHm2"].value

        if CpTh_value is not None:
            Cp1 = pars["Cp1"].value
            CpTh = CpTh_value
        else:
            Cp1 = 0.0
            CpTh = 0.0

        p1_Ns = np.array([pars[f"p1N_{j}"].value for j in range(n_signals)])
        p1_Us = np.array([pars[f"p1U_{j}"].value for j in range(n_signals)])
        intercepts_intermediates = np.array([pars[f"bI_{j}"].value for j in range(n_signals)])

        if baseline_native_params[0]:
            p2_Ns = np.array([pars[f"p2N_{j}"].value for j in range(n_signals)])
        else:
            p2_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[0]:
            p2_Us = np.array([pars[f"p2U_{j}"].value for j in range(n_signals)])
        else:
            p2_Us = np.zeros(n_signals)

        if baseline_native_params[1]:
            p3_Ns = np.array([pars[f"p3N_{j}"].value for j in range(n_signals)])
        else:
            p3_Ns = np.zeros(n_signals)

        if baseline_unfolded_params[1]:
            p3_Us = np.array([pars[f"p3U_{j}"].value for j in range(n_signals)])
        else:
            p3_Us = np.zeros(n_signals)

        if model_scale_factor:
            n_fit_factors = nr_olig - len(scale_factor_exclude_ids)
            factors = np.array([pars[f"sf_{j}"].value for j in range(n_fit_factors)])
            for id_ex in scale_factor_exclude_ids:
                factors = np.insert(factors, id_ex, 1.0)
            factors = np.tile(factors, n_signals)

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p1_N = p1_Ns[signal_ids[i]]
            p1_U = p1_Us[signal_ids[i]]
            intercepts_intermediate = intercepts_intermediates[signal_ids[i]]
            p2_N = p2_Ns[signal_ids[i]]
            p2_U = p2_Us[signal_ids[i]]
            p3_N = p3_Ns[signal_ids[i]]
            p3_U = p3_Us[signal_ids[i]]

            c = oligomer_concentrations[i]


            y = signal_fx(
                T, c, Tm1, DHm1, Tm2, DHm2,
                0, p1_N, p2_N, p3_N,
                0, p1_U, p2_U, p3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                intercepts_intermediate,
                Cp1,CpTh,
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        return np.concatenate(signal, axis=0)

    def residuals(pars):
        return model(pars) - all_signal

    if method == "least_sq":
        lmfit_method = "least_squares"
    elif method == "curve_fit":
        lmfit_method = "least_squares"
    else:
        lmfit_method = method

    minimizer = lmfit.Minimizer(residuals, params_lmfit)
    result = minimizer.minimize(method=lmfit_method)

    global_fit_params = np.array([result.params[name].value for name in param_names])

    if result.covar is not None:
        cov = result.covar
    else:
        J = result.jac
        dof = len(all_signal) - len(global_fit_params)
        residual_variance = np.sum(result.residual**2) / max(dof, 1)
        cov = pinv(J.T @ J) * residual_variance

    predicted = model(result.params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    # Convert the Tm to Celsius
    global_fit_params[0] = temperature_to_celsius(global_fit_params[0])
    global_fit_params[2] = temperature_to_celsius(global_fit_params[2])

    return global_fit_params, cov, predicted_lst, result, minimizer

def evaluate_need_to_refit(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        fit_m1=False,
        check_cp=True,
        check_dh=True,
        check_tm=True,
        fixed_cp=False,
        threshold=0.05,
        fit_m0=True
    ):

    """
    Check and expand parameter bounds when fitted parameters are too close to boundaries.

    Parameters
    ----------
    global_fit_params : array-like
        Fitted parameters
    high_bounds : array-like
        Upper bounds
    low_bounds : array-like
        Lower bounds
    p0 : array-like
        Initial guess for parameters
    fit_m1 : bool, optional
        Whether m1 (temperature dependence of m-value) is fitted
    check_cp, check_dh, check_tm : bool, optional
        Whether to check boundaries for Cp, DHm, and Tm respectively
    fixed_cp : bool, optional
        Whether the Cp value is fixed
    threshold : float, optional
        Threshold to compare if the fitted parameters are too close to the boundaries
    fit_m0 : bool, optional
        Whether m0 (m-value) is fitted (not in oligomeric models)

    Returns
    -------
    re_fit : bool
        True if a refit is recommended after bounds expansion
    p0 : array-like
        Updated initial parameters
    low_bounds : array-like
        Updated lower bounds
    high_bounds : array-like
        Updated upper bounds
    """

    # We need to create copies of the arrays, otherwise they will be overwritten
    global_fit_params = global_fit_params.copy()
    p0 = p0.copy()
    high_bounds = high_bounds.copy()
    low_bounds = low_bounds.copy()

    re_fit = False

    # Check the Tm boundary - upper
    tm_diff = high_bounds[0] - global_fit_params[0]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        high_bounds[0] = global_fit_params[0] + 12
        p0[0] = global_fit_params[0] + 5
        re_fit = True

    # Check the Tm boundary - lower
    tm_diff = global_fit_params[0] - low_bounds[0]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        low_bounds[0] = global_fit_params[0] - 12
        p0[0] = global_fit_params[0] - 5
        re_fit = True

    # Check the Dh boundary
    dh_diff = high_bounds[1] - global_fit_params[1]
    # Expand the boundary if the Dh is too close to the boundary
    if dh_diff < 20 and check_dh:
        high_bounds[1] = global_fit_params[1] + 80
        p0[1] = global_fit_params[1] + 50
        re_fit = True

    id_next = 2
    if not fixed_cp:

        # Check the Cp boundary
        cp_diff = high_bounds[2] - global_fit_params[2]
        # Expand the boundary if the Cp is too close to the boundary
        if cp_diff < 0.25 and check_cp:
            high_bounds[2] = global_fit_params[2] + 1
            p0[2] = global_fit_params[2] + 0.5
            re_fit = True
        
        id_next += 1

    if fit_m0:
        # Check the m-value boundary
        m_diff = high_bounds[id_next] - global_fit_params[id_next]
        # Expand the boundary if the m-value is too close to the boundary
        if m_diff < 0.5:
            high_bounds[id_next] = global_fit_params[id_next] + 2
            p0[id_next] = global_fit_params[id_next] + 0.5
            re_fit = True

        # Evaluate if m1 is fitted
        id_start = id_next + 1
    else:
        id_start = id_next


    if fit_m1:

        m1_diff = high_bounds[id_start] - global_fit_params[id_start]
        # Expand the boundary if the m-value is too close to the boundary
        if m1_diff < 0.1:
            high_bounds[id_start] = global_fit_params[id_start] + 1
            re_fit = True

        m1_diff = global_fit_params[id_start] - low_bounds[id_start]
        # Expand the boundary if the m-value is too close to the boundary
        if m1_diff < 0.1:
            low_bounds[id_start] = global_fit_params[id_start] - 1
            re_fit = True

        id_start += 1

    difference_to_upper = np.array([np.abs((a-b)/a) if a != np.inf and a != 0  else np.inf for a, b in zip(high_bounds[id_start:], global_fit_params[id_start:])])
    difference_to_lower = np.array([np.abs((a-b)/a) if b != -np.inf and a != 0 else np.inf for a, b in zip(global_fit_params[id_start:], low_bounds[id_start:])])

    # Evaluate all the other parameters
    for i in (range(len(global_fit_params)-id_start)):

        diff_to_high_i = difference_to_upper[i]
        diff_to_low_i = difference_to_lower[i]

        if diff_to_high_i < threshold:

            value = high_bounds[i+id_start]

            high_bounds[i+id_start] = value * 50 if value > 0 else value / 50
            re_fit = True

        if diff_to_low_i < threshold:

            value = low_bounds[i+id_start]
            low_bounds[i+id_start] = value * 50 if value < 0 else value / 50
            re_fit = True

    return re_fit, p0, low_bounds, high_bounds


def evaluate_need_to_refit_three_state(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        check_dh=True,
        check_tm=True,
        given_cp=False,
        threshold=0.05,
):
    """
    Check and expand parameter bounds when fitted parameters are too close to boundaries or if T1 is smaller than T2

    Parameters
    ----------
    global_fit_params : array-like
        Fitted parameters
    high_bounds : array-like
        Upper bounds
    low_bounds : array-like
        Lower bounds
    p0 : array-like
        Initial guess for parameters
    check_dh, check_tm : bool, optional
        Whether to check boundaries for DHms, and Tms respectively
    given_cp : bool, optional
        If a CpTh value is given the refitting structure needs to be adjusted. Currently there is no refitting when CpTh
        is given
    threshold : float, optional
        Threshold to compare if the fitted parameters are too close to the boundaries

    Returns
    -------
    re_fit : bool
        True if a refit is recommended after bounds expansion
    p0 : array-like
        Updated initial parameters
    low_bounds : array-like
        Updated lower bounds
    high_bounds : array-like
        Updated upper bounds
    """

    # We need to create copies of the arrays, otherwise they will be overwritten
    global_fit_params = global_fit_params.copy()
    p0 = p0.copy()
    high_bounds = high_bounds.copy()
    low_bounds = low_bounds.copy()

    re_fit = False

    # Check Tms

    # Check Tm1 is valid
    if global_fit_params[0] < 0:
        low_bounds[0] = 20
        p0[0] = 30
        high_bounds[0] = 90
        re_fit = True

    # Check the Tm1 boundary - upper
    tm_diff = high_bounds[0] - global_fit_params[0]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        high_bounds[0] = global_fit_params[0] + 12
        p0[0] = global_fit_params[0] + 5
        re_fit = True

    # Check the Tm1 boundary - lower
    tm_diff = global_fit_params[0] - low_bounds[0]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        low_bounds[0] = max(global_fit_params[0] - 12, 271)
        p0[0] = global_fit_params[0] - 5
        re_fit = True

    # Check Tm2 is valid
    if global_fit_params[2] < 0:
        low_bounds[2] = 20
        p0[2] = 30
        high_bounds[2] = 380
        re_fit = True

    # Check the Tm2 boundary - upper
    tm_diff = high_bounds[2] - global_fit_params[2]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        high_bounds[2] = global_fit_params[2] + 12
        p0[2] = global_fit_params[2] + 5
        re_fit = True

    # Check the Tm2 boundary - lower
    tm_diff = global_fit_params[2] - low_bounds[2]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        low_bounds[2] = max(global_fit_params[2] - 12, 271)
        p0[2] = global_fit_params[2] - 5
        re_fit = True

    # Check Tm1 smaller than Tm2

    if global_fit_params[0] > global_fit_params[2]:
        mid_point = (global_fit_params[0] + global_fit_params[2]) / 2

        p0[0] = mid_point - 10
        high_bounds[0] = mid_point + 5
        low_bounds[0] = p0[0] - 12

        p0[2] = mid_point + 10
        high_bounds[2] = p0[2] + 12
        low_bounds[2] = mid_point - 5

        re_fit = True

    # Check DHs

    # Check the Dh1 boundary
    dh_diff = high_bounds[1] - global_fit_params[1]
    # Expand the boundary if the Dh is too close to the boundary
    if dh_diff < 20 and check_dh:
        high_bounds[1] = global_fit_params[1] + 80
        p0[1] = global_fit_params[1] + 50
        re_fit = True

    # Check the Dh2 boundary
    dh_diff = high_bounds[3] - global_fit_params[3]
    # Expand the boundary if the Dh is too close to the boundary
    if dh_diff < 20 and check_dh:
        high_bounds[3] = global_fit_params[3] + 80
        p0[3] = global_fit_params[3] + 50
        re_fit = True

    id_start = 4

    # Adjust data for Cp1
    if given_cp:
        id_start += 1

    difference_to_upper = np.array([np.abs((a - b) / a) if a != np.inf and a != 0 else np.inf for a, b in
                                    zip(high_bounds[id_start:], global_fit_params[id_start:])])
    difference_to_lower = np.array([np.abs((a - b) / a) if b != -np.inf and a != 0 else np.inf for a, b in
                                    zip(global_fit_params[id_start:], low_bounds[id_start:])])

    # Evaluate all the other parameters
    for i in (range(len(global_fit_params) - id_start)):

        diff_to_high_i = difference_to_upper[i]
        diff_to_low_i = difference_to_lower[i]

        if diff_to_high_i < threshold:
            value = high_bounds[i + id_start]

            high_bounds[i + id_start] = value * 50 if value > 0 else value / 50
            re_fit = True

        if diff_to_low_i < threshold:
            value = low_bounds[i + id_start]
            low_bounds[i + id_start] = value * 50 if value < 0 else value / 50
            re_fit = True

    return re_fit, p0, low_bounds, high_bounds

def evaluate_fitting_and_refit(
        global_fit_params,
        cov,
        predicted,
        high_bounds,
        low_bounds,
        p0,
        fit_m_dep,
        limited_cp,
        limited_dh,
        limited_tm,
        fixed_cp,
        kwargs,
        fit_fx,
        n = 3,
        threshold=0.05,
        fit_m_value=True,
        three_state_model=False):

    """
    Evaluate if the fitted parameters are too close to the fitting boundaries.
    If they are, re-fit with new expanded boundaries

    Parameters
    ----------
    global_fit_params: array-like
        fitted parameters
    cov: array-like
        covariance matrix of the fitted parameters
    predicted: list
        list of lists with the fitted values
    high_bounds: array-like
        upper bounds of the fitting parameters
    low_bounds: array-like
        lower bounds of the fitting parameters
    p0: array-like
        initial guess for the fitting parameters
    fit_m_dep: boolean
        if the m-dependence on temperature is fitted
    limited_cp: boolean
        if the cp bounds are user-defined
    limited_dh: boolean
        if the DH bounds are user-defined
    limited_tm: boolean
        if the Tm values are user-defined
    fixed_cp: boolean
        if the cp value is fixed
    kwargs: dict
        dictionary with the arguments for the fitting function
    fit_fx: callable
        function to perform the fitting
    n: int, optional
        number of times to re-fit
    threshold : float, optional
        Threshold to compare if the fitted parameters are too close to the boundaries
    fit_m_value : bool, optional
        Whether m0 (m-value) is fitted (not in oligomeric models)
    three_state_model : bool, optional
        If a three state model is used different parameters are fitted

    Returns
    -------
    global_fit_params: array-like
        fitted parameters
    cov: array-like
        covariance matrix of the fitted parameters
    predicted: list
        list of lists with the fitted values
    p0: array-like
        initial guess for the fitting parameters
    low_bounds: array-like
        lower bounds of the fitting parameters
    high_bounds: array-like
        higher bounds of the fitting parameters
    """

    for _ in range(n):
        if three_state_model:
            re_fit, p0_new, low_bounds_new, high_bounds_new = evaluate_need_to_refit_three_state(
                global_fit_params,
                high_bounds,
                low_bounds,
                p0,
                check_dh=not limited_dh,
                check_tm=not limited_tm,
                given_cp=fixed_cp,
                threshold=threshold,
            )

        else:
            re_fit, p0_new, low_bounds_new, high_bounds_new = evaluate_need_to_refit(
                global_fit_params,
                high_bounds,
                low_bounds,
                p0,
                fit_m1=fit_m_dep,
                check_cp=not limited_cp,
                check_dh=not limited_dh,
                check_tm=not limited_tm,
                fixed_cp=fixed_cp,
                threshold=threshold,
                fit_m0=fit_m_value,
            )

        if re_fit:

            p0, low_bounds, high_bounds = p0_new, low_bounds_new, high_bounds_new

            kwargs['initial_parameters'] = p0
            kwargs['low_bounds'] = low_bounds
            kwargs['high_bounds'] = high_bounds

            global_fit_params, cov, predicted, result, minimizer = fit_fx(**kwargs)

        else:

            break

    return global_fit_params, cov, predicted, p0, low_bounds, high_bounds