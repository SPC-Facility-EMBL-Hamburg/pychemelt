import numpy as np
import pytest
import json

from pychemelt.utils.processing import (
    guess_Tm_from_derivative,
    get_colors_from_numeric_values,
    fit_local_thermal_unfolding_to_signal_lst,
    estimate_signal_baseline_params,
    adjust_value_to_interval,
    parse_number,
    are_all_strings_numeric,
    is_float,
    transform_to_list,
    set_condition_indexes_to_false,
    JsonEncoder
)


from pychemelt.utils.palette import VIRIDIS

def test_error_not_enough_data():

    temp_lst = [[1,2,3]]
    deriv_lst = [[1,2,3]]

    x1 = 0
    x2 = 5

    with pytest.raises(ValueError):

        guess_Tm_from_derivative(temp_lst, deriv_lst, x1, x2)

def test_get_colors_from_numeric_values():

    y = [1, 2, 3, 4, 5]

    colors = get_colors_from_numeric_values(y, 1, 5,use_log_scale=True)

    assert colors[0] == VIRIDIS[0]
    assert colors[-1] == VIRIDIS[-1]

    colors = get_colors_from_numeric_values(y, 1, 5,use_log_scale=False)

    assert colors[0] == VIRIDIS[0]
    assert colors[-1] == VIRIDIS[-1]

def test_trigger_exception_fit_local_thermal_unfolding_to_signal_lst():

    signal_lst = [[np.nan for _ in range(5)]]
    temp_lst = [[x for x in range(5)]]

    Tms, dHs, predicted_lst =  fit_local_thermal_unfolding_to_signal_lst(
        signal_lst, temp_lst, [100],
        [1], [1], [1], [1], [1], [1],
        baseline_native_fx = lambda x: x,
        baseline_unfolded_fx = lambda x: x
    )

    assert Tms == []

def test_adjust_value_to_interval():

    assert adjust_value_to_interval(10, 0, 100,1) == 10

    assert adjust_value_to_interval(-1, 0, 100,0.5) == 0.5

    assert adjust_value_to_interval(110, 0, 100,0.5) == 99.5

def test_parse_number():

    assert parse_number('10') == 10

    assert parse_number('10.5') == 10.5

    assert parse_number('1e-3') == 0.001

    with pytest.raises(ValueError):
        parse_number('not_a_number')

def test_are_all_strings_numeric():

    assert are_all_strings_numeric(['10', '20', '30']) == True

    assert are_all_strings_numeric(['10', '20.5', '1e-3']) == True

    assert are_all_strings_numeric(['10', 'not_a_number', '30']) == False

def test_is_float():

    assert is_float('10') == True

    assert is_float('10.5') == True

    assert is_float('1e-3') == True

    assert is_float('not_a_number') == False

def test_transform_to_list():

    assert transform_to_list("hola") == ["hola"]

    assert transform_to_list([1,2,3]) == [1,2,3]

    assert transform_to_list('string') == ['string']

    # none should be returned as none, not as a list with one element
    assert transform_to_list(None) == None

    # raise error if input is not string, list, float, int or None
    with pytest.raises(ValueError):
        transform_to_list({'key': 'value'})

def test_estimate_signal_baseline_params_tuple_windows():

    temp = np.arange(0.0, 10.0)
    signal = 2.0 * temp + 1.0

    p1Ns, p1Us, p2Ns, p2Us, p3Ns, p3Us = estimate_signal_baseline_params(
        signal_lst=[signal],
        temp_lst=[temp],
        native_baseline_type='linear',
        unfolded_baseline_type='linear',
        window_range_native=(2.0, 4.0),
        window_range_unfolded=(6.0, 8.0)
    )

    assert len(p1Ns) == 1
    assert len(p1Us) == 1
    assert len(p2Ns) == 1
    assert len(p2Us) == 1
    assert len(p3Ns) == 0
    assert len(p3Us) == 0

    np.testing.assert_allclose(p2Ns[0], 2.0, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(p2Us[0], 2.0, rtol=1e-3, atol=1e-6)

def test_estimate_signal_baseline_params_invalid_tuple_window():

    temp = np.arange(0.0, 10.0)
    signal = temp.copy()

    with pytest.raises(ValueError):
        estimate_signal_baseline_params(
            signal_lst=[signal],
            temp_lst=[temp],
            native_baseline_type='constant',
            unfolded_baseline_type='constant',
            window_range_native=(2.0,),
            window_range_unfolded=3.0
        )

def test_estimate_signal_baseline_params_invalid_tuple_order():

    temp = np.arange(0.0, 10.0)
    signal = temp.copy()

    with pytest.raises(ValueError, match='high > low'):
        estimate_signal_baseline_params(
            signal_lst=[signal],
            temp_lst=[temp],
            native_baseline_type='constant',
            unfolded_baseline_type='constant',
            window_range_native=(5.0, 5.0),
            window_range_unfolded=3.0
        )

def test_estimate_signal_baseline_params_invalid_scalar_width():

    temp = np.arange(0.0, 10.0)
    signal = temp.copy()

    with pytest.raises(ValueError, match='width must be > 0'):
        estimate_signal_baseline_params(
            signal_lst=[signal],
            temp_lst=[temp],
            native_baseline_type='constant',
            unfolded_baseline_type='constant',
            window_range_native=0.0,
            window_range_unfolded=3.0
        )

def test_estimate_signal_baseline_params_window_with_no_points():

    temp = np.arange(0.0, 10.0)
    signal = temp.copy()

    with pytest.raises(ValueError, match='No temperature points found'):
        estimate_signal_baseline_params(
            signal_lst=[signal],
            temp_lst=[temp],
            native_baseline_type='constant',
            unfolded_baseline_type='constant',
            window_range_native=(20.0, 30.0),
            window_range_unfolded=3.0
        )

def test_set_condition_indexes_to_false():

    conditions = [True, True, True, True]

    set_condition_indexes_to_false(conditions, ["1", "3"])

    assert conditions == [False, True, False, True]

    conditions = [True, True, True, True, True]

    set_condition_indexes_to_false(conditions, ["1-3"])

    assert conditions == [False, False, False, True, True]
    
class Unsupported:
    pass

def test_json_encoder_unsupported_type():
    with pytest.raises(TypeError, match="Object of type Unsupported is not JSON serializable"):
        json.dumps({"x": Unsupported()}, cls=JsonEncoder)