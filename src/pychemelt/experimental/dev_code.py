def solve_one_root_quadratic(a,b,c):
    """
    Obtain one root of the quadratic equation of the form ax^2 + bx + c = 0.

    Parameters
    ----------
    a : float
        Coefficient of x^2
    b : float
        Coefficient of x
    c : float
        Constant term

    Returns
    -------
    float
        One root of the quadratic equation
    """
    return 2*c / (-b - np.sqrt(b**2 - 4*a*c))


def solve_one_root_depressed_cubic(p,q):

    """
    Obtain one root of the depressed cubic equation of the form x^3 + p x + q = 0.

    Parameters
    ----------
    p : float
        Coefficient of x
    q : float
        Constant term

    Returns
    -------
    float
        One real root of the cubic equation
    """

    delta = np.sqrt((q**2/4) + (p**3/27))

    return np.cbrt(-q/2+delta) + np.cbrt(-q/2-delta)

def residuals_squares_sum(y_true,y_pred):

    """
    Calculate the residual sum of squares.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Residual sum of squares
    """

    # Convert to numpy arrays if it is a list
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    rss = np.sum((y_true - y_pred)**2)

    return rss



def r_squared(y_true, y_pred):
    """
    Calculate the R-squared value for a regression model.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        R-squared value
    """

    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_total


def adjusted_r2(r2, n, p):
    """
    Calculate the adjusted R-squared value for a regression model.

    Parameters
    ----------
    r2 : float
        R-squared value
    n : int
        Number of observations
    p : int
        Number of predictors

    Returns
    -------
    float
        Adjusted R-squared value
    """

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def compute_aic(y_true, y_pred, k):
    """
    Compute the Akaike Information Criterion (AIC) for a regression model.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    k : int
        Number of parameters in the model

    Returns
    -------
    float
        AIC value
    """

    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k


def compare_akaikes(akaikes_1, akaikes_2, akaikes_3, akaikes_4, denaturant_concentrations):
    model_names = ['Linear - Linear', 'Linear - Quadratic',
                   'Quadratic - Linear', 'Quadratic - Quadratic']

    akaikes_df = pd.DataFrame({
        'Model': model_names})

    i = 0
    for a1, a2, a3, a4 in zip(akaikes_1, akaikes_2, akaikes_3, akaikes_4):
        # Create a new column with the Akaike values
        # The name is the denaturant concentration

        # Compute delta AIC
        min_aic = np.min([a1, a2, a3, a4])
        a1 = a1 - min_aic
        a2 = a2 - min_aic
        a3 = a3 - min_aic
        a4 = a4 - min_aic

        akaikes_df[str(i) + '_' + str(denaturant_concentrations[i])] = [a1, a2, a3, a4]
        i += 1

    # Find the best model for each denaturant concentration
    best_models_ids = []
    for i in range(len(denaturant_concentrations)):

        # Get the column with the Akaike values
        aic_col = akaikes_df.iloc[:, i + 1].to_numpy()

        # Find index that sort them from min to max a numpy array
        sorted_idx = np.argsort(aic_col)

        first_model_id = np.arange(4)[sorted_idx][0]
        second_model_id = np.arange(4)[sorted_idx][1]
        third_model_id = np.arange(4)[sorted_idx][2]
        fourth_model_id = np.arange(4)[sorted_idx][3]

        best_models_ids.append(first_model_id)

        # Compare the AIC value of the second model to the first one
        if aic_col[second_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(second_model_id)

        # Compare the AIC value of the third model to the first one
        if aic_col[third_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(third_model_id)

        # Compare the AIC value of the fourth model to the first one
        if aic_col[fourth_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(fourth_model_id)

    # Print the overall best model
    best_model_all = Counter(best_models_ids).most_common(1)[0][0]
    return model_names[best_model_all]


def rss_p(rrs0, n, p, alfa):

    """
    Given the residuals of the best fitted model,
    compute the desired residual sum of squares for a 1-alpha confidence interval.
    This is used to compute asymmetric confidence intervals for the fitted parameters.

    Parameters
    ----------
    rrs0 : float
        Residual sum of squares of the model with the best fit
    n : int
        Number of data points
    p : int
        Number of parameters
    alfa : float
        Desired significance level (alpha)

    Returns
    -------
    float
        Residual sum of squares for the desired confidence interval
    """

    critical_value = stats.f.ppf(q=1 - alfa, dfn=1, dfd=n - p)

    return rrs0 * (1 + critical_value / (n - p))


def get_desired_rss(y, y_fit, p,alpha=0.05):

    """
    Given the observed and fitted data, find the residual sum of squares required for a 1-alpha confidence interval.

    Parameters
    ----------
    y : array-like
        Observed values or list of arrays
    y_fit : array-like
        Fitted values or list of arrays
    p : int
        Number of parameters
    alpha : float, optional
        Desired significance level (default: 0.05)

    Returns
    -------
    float
        Residual sum of squares corresponding to the desired confidence interval
    """

    # If y is of type list, convert it to a numpy array by concatenating
    if isinstance(y, list):
        y = np.concatenate(y,axis=0)
    # If y_fit is of type list, convert it to a numpy array by concatenating
    if isinstance(y_fit, list):
        y_fit = np.concatenate(y_fit,axis=0)

    n = len(y)

    rss = get_rss(y, y_fit)

    return rss_p(rss, n, p, alpha)

def compare_linear_to_quadratic(x,y):

    """
    Compare the linear and quadratic fits to the data using an F-test.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    bool
        True if the linear model is statistically preferable to the quadratic model
    """

    m, b       = fit_line_robust(x, y)
    y_pred_lin = m * x + b

    a,b,c     = fit_quadratic_robust(x, y)
    y_pred_quad = a * x ** 2 + b * x + c

    # Residual sums
    rss_lin = np.sum((y - y_pred_lin) ** 2)
    rss_quad = np.sum((y - y_pred_quad) ** 2)

    # R² and Adjusted R²
    n = len(x)
    p_lin = 1
    p_quad = 2

    # F-test
    numerator   = (rss_lin - rss_quad) / (p_quad - p_lin)
    denominator = rss_quad / (n - (p_quad + 1))
    f_stat = numerator / denominator
    p_value = 1 - f_dist.cdf(f_stat, dfn=p_quad - p_lin, dfd=n - (p_quad + 1))

    # True if linear model is better
    return p_value > 0.05

def fu_two_state_dimer(K,C):
    """
    Given the equilibrium constant K of N2 <-> 2U and the concentration of dimer equivalent C,
    return the fraction of unfolded protein.

    Parameters
    ----------
    K : float
        Equilibrium constant of the reaction N2 <-> 2U
    C : float
        Concentration of dimer equivalent

    Returns
    -------
    float
        Fraction of unfolded protein
    """

    return solve_one_root_quadratic(4*C, K, -K)

def arrhenius(T, Tf, Ea):
    """
    Arrhenius equation: defines dependence of reaction rate constant k on temperature.
    In this version of the equation we use Tf (a temperature of k=1) to avoid specifying a pre-exponential constant A.

    Parameters
    ----------
    T : array-like
        Temperature (°C or K)
    Tf : float
        Reference temperature at which the reaction rate constant equals 1 (°C or K)
    Ea : float
        Activation energy (kcal/mol)

    Returns
    -------
    numpy.ndarray
        Reaction rate constant at the given temperature
    """

    T  = temperature_to_kelvin(T)
    Tf = temperature_to_kelvin(Tf)

    return np.exp(-Ea / R_gas * (1 / T - 1 / Tf))


def fit_tc_unfolding_many_signals_slow(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        fit_slope_native_temp=True,
        fit_slope_unfolded_temp=True,
        fit_slope_native_den=True,
        fit_slope_unfolded_den=True,
        fit_quadratic_native=False,
        fit_quadratic_unfolded=False,
        oligomer_concentrations=None,
        fit_m1=False,
        model_scale_factor=False,
        scale_factor_exclude_ids=[]):
    """
    Fit thermochemical unfolding curves for many signals (slow variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
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
    fit_slope_native_temp : bool, optional
        Whether to fit the temperature slope of the native baseline (per-signal)
    fit_slope_unfolded_temp : bool, optional
        Whether to fit the temperature slope of the unfolded baseline (per-signal)
    fit_slope_native_den : bool, optional
        Whether to fit the denaturant slope of the native baseline (per-signal)
    fit_slope_unfolded_den : bool, optional
        Whether to fit the denaturant slope of the unfolded baseline (per-signal)
    fit_quadratic_native : bool, optional
        Whether to fit a quadratic temperature term for the native baseline (per-signal)
    fit_quadratic_unfolded : bool, optional
        Whether to fit a quadratic temperature term for the unfolded baseline (per-signal)
    oligomer_concentrations : list, optional
        Oligomer concentrations per dataset (used by oligomeric models)
    fit_m1 : bool, optional
        Whether to include and fit temperature dependence of the m-value (m1)
    model_scale_factor : bool, optional
        If True, include a per-denaturant concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1 (useful to avoid fitting trivial factors)

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

    n_signals = np.max(signal_ids) + 1

    nr_den = int(len(denaturant_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    # Find if highest concentration of denaturant has a higher signal or not
    if model_scale_factor:
        den_conc_simple = denaturant_concentrations[:nr_den]

        # Find the index that sorts them in descending order from highest to lowest
        sort_indeces = np.argsort(den_conc_simple)[::-1]

        signal_first = list_of_signals[:nr_den]

        signal_sort = [signal_first[i] for i in sort_indeces]

        higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]

    def unfolding(dummyVariable, *args):

        Tm, DHm, Cp0, m0 = args[:4]  # Enthalpy of unfolding, Temperature of melting, Cp0, m0, m1

        id_param_init = 4 + fit_m1
        m1 = args[4] if fit_m1 else 0

        # First filter, verify that DG is not lower than 0 at 5C
        # In other words, we do not have cold denaturation at 5C
        """
        Tfive = temperature_to_kelvin(5)
        TmK   = temperature_to_kelvin(Tm)

        DGfive = DHm * (1 - Tfive / TmK) + Cp0 * (Tfive - TmK - Tfive * np.log(Tfive / TmK))

        if DGfive < 0:

            return np.zeros(len(all_signal))
        """

        a_Ns = args[id_param_init:id_param_init + n_signals]
        a_Us = args[id_param_init + n_signals:id_param_init + 2 * n_signals]

        id_param_init = id_param_init + 2 * n_signals
        if fit_slope_native_temp:
            b_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            b_Ns = [0] * n_signals

        if fit_slope_unfolded_temp:
            b_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            b_Us = [0] * n_signals

        if fit_slope_native_den:
            c_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            c_Ns = [0] * n_signals

        if fit_slope_unfolded_den:
            c_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            c_Us = [0] * n_signals

        if fit_quadratic_native:
            d_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            d_Ns = [0] * n_signals

        if fit_quadratic_unfolded:
            d_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            d_Us = [0] * n_signals

        if model_scale_factor:
            # One per denaturant concentration
            factors = args[id_param_init:id_param_init + (nr_den - len(scale_factor_exclude_ids))]

            for id_ex in scale_factor_exclude_ids:
                factors = np.insert(factors, id_ex, 1)

            # Repeat the list so have the same length as list_of_temperatures, equal to denaturant concentration * number of signals
            factors = np.tile(factors, n_signals)

            id_param_init += nr_den

        signal = []

        for i, T in enumerate(list_of_temperatures):
            a_N = a_Ns[signal_ids[i]]
            b_N = b_Ns[signal_ids[i]]
            c_N = c_Ns[signal_ids[i]]
            d_N = d_Ns[signal_ids[i]]

            a_U = a_Us[signal_ids[i]]
            b_U = b_Us[signal_ids[i]]
            c_U = c_Us[signal_ids[i]]
            d_U = d_Us[signal_ids[i]]

            d = denaturant_concentrations[i]

            c = 0 if oligomer_concentrations is None else oligomer_concentrations[i]

            d_factor = 1

            d = d * d_factor

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                a_N, b_N, c_N, d_N,
                a_U, b_U, c_U, d_U, c
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        # Second filter, verify that higher_den_equal_higher_signal is same in the raw and fitted signal
        if model_scale_factor:

            signal_first = signal[:nr_den]

            signal_sort = [signal_first[i] for i in sort_indeces]

            pred_higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]

            if pred_higher_den_equal_higher_signal != higher_den_equal_higher_signal:
                return np.zeros(len(all_signal))

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds))

    predicted = unfolding(1, *global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n


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

    return global_fit_params, cov, predicted_lst