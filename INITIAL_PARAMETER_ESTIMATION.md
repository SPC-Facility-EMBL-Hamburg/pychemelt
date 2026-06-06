# Initial Parameter Estimation for Global Thermal Unfolding Fitting

## Overview

Before calling `fit_thermal_unfolding_global()`, the pychemelt package employs a systematic multi-step workflow to estimate initial parameter values. This document explains the complete process, which ensures that the global fitting routine starts with reasonable parameter guesses, leading to more robust and reliable fits.

---

## Complete Workflow

The typical workflow involves these steps in sequence:

1. **Data Loading and Signal Selection**
2. **Condition Selection**
3. **Temperature Range Setting**
4. **Derivative Estimation**
5. **Initial Tm Guessing**
6. **Baseline Parameter Estimation**
7. **Local Curve Fitting**
8. **Heat Capacity (Cp) Estimation**
9. **Global Fitting** ← The final step

The method `guess_initial_parameters()` orchestrates most of these steps automatically.

---

## Detailed Step-by-Step Process

### 1. Data Loading and Signal Selection

**Methods involved:**
- `read_file()` or `read_files()`
- `set_signal(signal_names)`
- `set_denaturant_concentrations(concentrations)`

**What happens:**
- Raw experimental data is loaded from files (Excel, CSV, etc.)
- Multiple signals can be selected (e.g., `['330 nm', '350 nm']`)
- Each signal has associated temperature and signal intensity arrays
- Denaturant concentrations are assigned to each experimental condition
- NaN values are automatically removed from signal and temperature arrays

**Created attributes:**
- `signal_lst_pre_multiple`: List of signal arrays for each signal type
- `temp_lst_pre_multiple`: List of temperature arrays for each signal type
- `signal_names`: Names of selected signals
- `denaturant_concentrations_pre`: Array of denaturant concentrations

---

### 2. Condition Selection

**Method:** `select_conditions(boolean_lst, normalise_to_global_max=True)`

**What happens:**
- A boolean list filters which experimental conditions to include in the analysis
- Signals can be normalized to a global maximum across all conditions (default: True)
- The denaturant concentrations are filtered accordingly

**Example:**
```python
# Keep only the first 3 conditions out of 10
sample.select_conditions([True, True, True] + [False]*7)
```

**Created attributes:**
- `signal_lst_multiple`: Filtered signal lists
- `temp_lst_multiple`: Filtered temperature lists
- `denaturant_concentrations`: Filtered denaturant concentrations
- `denaturant_concentrations_expanded`: Flattened array matching expanded signal data
- `nr_den`: Number of denaturant concentrations

---

### 3. Temperature Range Setting

**Method:** `set_temperature_range(min_temp, max_temp)`

**What happens:**
- Restricts the temperature range for analysis
- Useful for excluding problematic regions at the start or end of melting curves

**Created attributes:**
- `user_min_temp`, `user_max_temp`: User-defined temperature limits

---

### 4. Derivative Estimation

**Method:** `estimate_derivative(window_length=5)`

**What happens:**
- Calculates the first derivative of signal with respect to temperature
- Uses Savitzky-Golay filtering for smooth derivative estimation
- Requires evenly spaced temperature data; if not, data is interpolated
- The derivative helps identify the melting transition

**Key function:** `first_derivative_savgol(x, y, window_length, polyorder=4)`

**Created attributes:**
- `deriv_lst_multiple`: List of derivative arrays for each signal
- `temp_deriv_lst_multiple`: Corresponding temperature arrays for derivatives

---

### 5. Initial Tm Guessing

**Method:** `guess_Tm(x1=6, x2=11)`

**What happens:**
- Identifies the melting temperature (Tm) from the derivative peak
- For each curve:
  1. Estimates baseline derivative values at the start (x1 degrees from min) and end (x2 degrees from max)
  2. Calculates median derivative in these baseline regions
  3. Finds the temperature where the derivative is maximum (or minimum)
  4. This temperature is the initial Tm guess

**Algorithm (from `guess_Tm_from_derivative`):**
```python
# For each melting curve:
# 1. Get baseline derivative at start and end
median_start = median(derivative[temp < min_temp + x1])
median_end = median(derivative[temp > max_temp - x2])

# 2. Find maximum derivative in the transition region
transition_region = temp[(temp > min_temp + x2) & (temp < max_temp - x1)]
max_derivative_idx = argmax(abs(derivative[transition_region]))
Tm_initial = temp[max_derivative_idx]
```

**Created attributes:**
- `t_melting_init_multiple`: List of initial Tm guesses for each signal and condition
- `t_melting_df_multiple`: DataFrames with Tm vs denaturant concentration

---

### 6. Baseline Parameter Estimation

**Method:** `estimate_baseline_parameters(native_baseline_type, unfolded_baseline_type, window_range_native=12, window_range_unfolded=12)`

**What happens:**
- Estimates baseline parameters for both native and unfolded states
- Uses data from the beginning and end of each curve
- Supports four baseline types: 'constant', 'linear', 'quadratic', 'exponential'

**Process (from `estimate_signal_baseline_params`):**

For each melting curve:

1. **Extract native baseline region:**
   ```python
   signal_native = signal[temp < min(temp) + window_range_native]
   temp_native = temp[temp < min(temp) + window_range_native]
   temp_native = shift_temperature(temp_native)  # Center at Tref (298.15 K)
   ```

2. **Extract unfolded baseline region:**
   ```python
   signal_unfolded = signal[temp > max(temp) - window_range_unfolded]
   temp_unfolded = temp[temp > max(temp) - window_range_unfolded]
   temp_unfolded = shift_temperature(temp_unfolded)
   signal_unfolded = signal_unfolded / oligomer_number  # Correct for oligomerization
   ```

3. **Fit baseline parameters based on type:**

   - **Constant:** `p1 = median(signal)`
   - **Linear:** Robust line fit → `p1 (intercept), p2 (slope)`
   - **Quadratic:** Robust quadratic fit → `p1, p2, p3`
   - **Exponential:** Robust exponential fit → `p1, p2, p3`

**Baseline function forms:**
- **Constant:** `baseline(T) = a`
- **Linear:** `baseline(T) = a + b·ΔT`
- **Quadratic:** `baseline(T) = a + b·ΔT + c·ΔT²`
- **Exponential:** `baseline(T) = a + c·exp(-α·ΔT)`

where `ΔT = T - Tref` (temperature shifted to reference)

**Created attributes:**
- `first_param_Ns_per_signal`, `first_param_Us_per_signal`: Intercepts (a)
- `second_param_Ns_per_signal`, `second_param_Us_per_signal`: Slopes/pre-exponential factors (b or c)
- `third_param_Ns_per_signal`, `third_param_Us_per_signal`: Quadratic/exponential coefficients
- `baseline_N_fx`, `baseline_U_fx`: Function references for native and unfolded baselines
- `native_baseline_type`, `unfolded_baseline_type`: Baseline type strings

---

### 7. Local Curve Fitting

**Method:** `fit_thermal_unfolding_local()`

**What happens:**
- Fits each individual melting curve independently
- Uses a two-state unfolding model with temperature-dependent thermodynamics
- Each curve has its own Tm and ΔH (enthalpy)
- Baseline parameters are fixed to the estimated values from step 6

**Fitting model (per curve):**
```python
# Two-state model:
# Signal = fU·baseline_unfolded + fN·baseline_native
# where fU = fraction unfolded, fN = fraction native
# fU depends on: Tm, ΔH, temperature
```

**Initial parameters for each curve:**
```python
p0 = [
    Tm_initial,           # From step 5
    85,                   # Default ΔH (kcal/mol)
    p1_native,            # From step 6
    p1_unfolded,          # From step 6
    p2_native (if applicable),
    p2_unfolded (if applicable),
    p3_native (if applicable),
    p3_unfolded (if applicable)
]
```

**Bounds:**
- Tm: `[min(temp), max(temp) + 15]`
- ΔH: `[10, 500]` kcal/mol
- Baseline parameters: Generous bounds around initial estimates

**Quality control:**
- Only fits with relative errors < 50% for Tm and ΔH are retained
- Poor fits are rejected

**Created attributes:**
- `Tms_multiple`: List of fitted Tm values for each curve
- `dHs_multiple`: List of fitted ΔH values for each curve
- `predicted_lst_multiple`: List of fitted signal predictions
- `single_fit_done`: Flag set to True

---

### 8. Heat Capacity (Cp) Estimation

**Method:** `guess_Cp()`

**What happens:**
- Estimates the heat capacity change (ΔCp) upon unfolding
- Uses the relationship between Tm and ΔH from the local fits
- The Kirchhoff relation states: ΔH(T) = ΔH(Tm) + ΔCp·(T - Tm)
- Therefore: slope of ΔH vs Tm plot ≈ ΔCp

**Algorithm:**

1. **Collect Tm and ΔH from local fits:**
   ```python
   Tms = [all Tm values from step 7]
   dHs = [all ΔH values from step 7]
   ```

2. **Robust linear regression:**
   ```python
   slope, intercept = fit_line_robust(Tms, dHs)
   ```

3. **Outlier detection and removal:**
   ```python
   outliers = find_line_outliers(slope, intercept, Tms, dHs, sigma=2.5)
   # Remove outliers and refit if necessary
   ```

4. **Estimate Cp:**
   ```python
   Cp0 = slope if slope > 0 else -1
   ```

5. **Sanity check against empirical formula:**
   ```python
   expected_Cp0 = n_residues × 0.0148 - 0.1267  # Empirical relation
   if Cp0 < expected_Cp0/1.5 or Cp0 > expected_Cp0×1.5:
       Cp0 = expected_Cp0
   ```

6. **Ensure Cp is positive:**
   ```python
   Cp0 = max(Cp0, 0)
   ```

**Created attributes:**
- `Tms`: Flattened array of all Tm values (outliers removed)
- `dHs`: Flattened array of all ΔH values (outliers removed)
- `slope_dh_tm`: Slope of ΔH vs Tm relationship
- `intercept_dh_tm`: Intercept of ΔH vs Tm relationship
- `Cp0`: Initial estimate of heat capacity change

---

## Summary of Initial Parameters for Global Fitting

When `fit_thermal_unfolding_global()` is finally called, it has access to:

### Thermodynamic Parameters:
1. **Tm** (initial): Maximum Tm from local fits
2. **ΔH** (initial): ΔH at maximum Tm (or 80 kcal/mol minimum)
3. **Cp** (initial): Estimated from ΔH vs Tm slope
4. **m-value** (initial): Fixed at 2.8 kcal/mol/M (empirical default)

### Baseline Parameters (for each signal × each denaturant concentration):
- **Native state:** p1 (intercept), p2 (slope/factor), p3 (quadratic/exponential)
- **Unfolded state:** p1 (intercept), p2 (slope/factor), p3 (quadratic/exponential)

### Parameter Bounds:
The global fitting uses bounds to constrain the optimization:

- **Tm bounds:**
  - Lower: `Tm_initial - 12°C`
  - Upper: `max(user_max_temp + 20, Tm_initial + 10)`

- **ΔH bounds:**
  - Lower: `10 kcal/mol` (or user-defined)
  - Upper: `500 kcal/mol` (or user-defined)

- **Cp bounds:**
  - Lower: `0.1 kcal/mol/°C`
  - Upper: `5 kcal/mol/°C`

- **m-value bounds:**
  - Lower: `0.5 kcal/mol/M`
  - Upper: `9 kcal/mol/M`

- **Baseline parameter bounds:** Set automatically based on initial estimates (typically ±200× the initial value)

---

## The `guess_initial_parameters()` Convenience Method

The `guess_initial_parameters()` method provides a streamlined workflow that:

1. Optionally switches to 'Ratio' signal if available (for better Tm estimation)
2. Calls `estimate_baseline_parameters()`
3. Calls `fit_thermal_unfolding_local()`
4. Calls `guess_Cp()`
5. Performs an initial `fit_thermal_unfolding_global()`
6. Stores the resulting thermodynamic parameters as `thermodynamic_params_guess`
7. Switches back to the original signal if step 1 was performed

This creates a set of refined initial parameters that can be used for subsequent fitting attempts with different models or baseline types.

---

## Why This Multi-Step Approach?

The hierarchical estimation strategy has several advantages:

1. **Robustness:** Starting with local fits ensures each curve contributes to the parameter estimation, even if some curves are noisy
2. **Physical constraints:** The Cp estimation uses the Kirchhoff relation, ensuring thermodynamic consistency
3. **Automation:** The entire process can run without user intervention
4. **Flexibility:** Users can override any step or provide custom initial values
5. **Quality control:** Outliers and poor fits are detected and excluded
6. **Convergence:** Good initial guesses dramatically improve the convergence of the global nonlinear fitting

---

## Example Usage

```python
from pychemelt import Monomer

# Create sample and load data
sample = Monomer()
sample.read_file('data.xlsx')
sample.set_denaturant_concentrations()
sample.set_signal(['330 nm', '350 nm'])
sample.select_conditions([True]*5 + [False]*5)  # Use first 5 conditions
sample.set_temperature_range(20, 90)

# Option 1: Manual step-by-step
sample.estimate_derivative()
sample.guess_Tm()
sample.estimate_baseline_parameters('linear', 'linear')
sample.fit_thermal_unfolding_local()
sample.guess_Cp()

# Now ready for global fitting
sample.fit_thermal_unfolding_global()

# Option 2: Automated with convenience method
sample.guess_initial_parameters(
    native_baseline_type='linear',
    unfolded_baseline_type='exponential',
    window_range_native=12,
    window_range_unfolded=12
)

# thermodynamic_params_guess is now set, ready for further fitting
sample.fit_thermal_unfolding_global(cp_limits=[0.2, 3.0])
```

---

## References

- **Kirchhoff relation:** ΔH(T) = ΔH(T₀) + ΔCp·(T - T₀)
- **Empirical Cp formula:** Robertson & Murphy (1997), *Chem. Rev.* 97, 1251-1267
- **Two-state unfolding model:** Standard thermodynamic model for protein folding

---

**Document created:** June 6, 2026  
**Package:** pychemelt  
**Author:** GitHub Copilot
