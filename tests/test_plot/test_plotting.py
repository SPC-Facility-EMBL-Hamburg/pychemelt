from random import sample

import plotly.graph_objs as go
import pytest

from pychemelt.utils.plotting import plot_unfolding, plot_baselines, plot_residuals
from pychemelt import Monomer as Sample

def test_plot_unfolding():

    sample = Sample()

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()
    sample.set_signal(['350nm', '330nm'])

    sample.select_conditions([True for _ in range(8)] + [False for _ in range(48 - 8)])

    fig = plot_unfolding(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    fig = plot_unfolding(sample, plot_derivative=True)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    sample.expand_multiple_signal()
    sample.estimate_baseline_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic'
    )
    sample.estimate_derivative()
    sample.guess_Tm()
    sample.n_residues = 130
    sample.guess_Cp()
    sample.set_signal_id()
    sample.fit_thermal_unfolding_local()
    sample.fit_thermal_unfolding_global()

    fig = plot_unfolding(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    fig = plot_unfolding(sample, plot_derivative=True)

    assert fig is not None
    assert isinstance(fig, go.Figure)

def test_plot_unfolding_with_baseline_df_and_scaled():

    sample = Sample()

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()
    sample.set_signal(['350nm', '330nm'])
    sample.max_points = 100

    sample.select_conditions([True for _ in range(8)] + [False for _ in range(48 - 8)])

    sample.expand_multiple_signal()
    sample.estimate_baseline_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic'
    )
    sample.estimate_derivative()
    sample.guess_Tm()
    sample.n_residues = 130
    sample.guess_Cp()
    sample.set_signal_id()
    sample.fit_thermal_unfolding_local()
    sample.fit_thermal_unfolding_global()
    sample.fit_thermal_unfolding_global_global()
    sample.fit_thermal_unfolding_global_global_global()

    sample.predict_baselines()

    fig = plot_unfolding(sample, plot_baseline_df=True)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    baseline_traces = [
        trace for trace in fig.data
        if trace.mode == 'lines'
        and trace.line is not None
        and trace.line.color == 'red'
        and trace.line.dash == 'dash'
    ]

    assert len(baseline_traces) == 2 * sample.nr_signals
    assert all(trace.showlegend is False for trace in baseline_traces)

    fig = plot_unfolding(sample, plot_baseline_df=False, use_scaled_data=True)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    # Raise value error if plot_baseline_df and use_scaled_data are both True
    with pytest.raises(ValueError):
        plot_unfolding(sample, plot_baseline_df=True, use_scaled_data=True)

def test_plot_baselines():
    sample = Sample()

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()
    sample.set_signal(['350nm', '330nm'])

    sample.select_conditions([True for _ in range(8)] + [False for _ in range(48 - 8)])

    sample.expand_multiple_signal()

    pytest.raises(ValueError, plot_baselines, sample)

    sample.estimate_baseline_parameters(
        native_baseline_type='constant',
        unfolded_baseline_type='quadratic'
    )

    fig = plot_baselines(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    sample.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    fig = plot_baselines(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

def test_plot_residuals():

    sample = Sample()

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()
    sample.set_signal(['350nm', '330nm'])

    sample.select_conditions([True for _ in range(8)] + [False for _ in range(48 - 8)])

    sample.expand_multiple_signal()
    sample.estimate_baseline_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic'
    )
    sample.estimate_derivative()
    sample.guess_Tm()
    sample.n_residues = 130
    sample.guess_Cp()
    sample.set_signal_id()

    # Raise value error if plot residuals is called before fitting
    with pytest.raises(ValueError):
        plot_residuals(sample)

    sample.fit_thermal_unfolding_local()
    sample.fit_thermal_unfolding_global()

    fig = plot_residuals(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    sample.denaturant_concentrations /= 1e12 # To force micromolar legend

    fig = plot_residuals(sample, individual_curves=True)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    # Count the number of traces in the individual curves plot
    num_traces = len(fig.data)
     # We have 8 conditions selected, two signals: 16 + 16 fitted curves = 32 traces in total
    assert num_traces == 32