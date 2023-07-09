from src.utils.training_utils import concat_dicts


def test_concat_basic():
    expected_result = {"a": [1, 3], "b": [2, 5], "c": [4], "d": [6]}
    dicts = [
        {"a": 1, "b": 2},
        {"a": 3, "c": 4},
        {"b": 5, "d": 6}
    ]
    assert concat_dicts(dicts) == expected_result


def test_concat_empty_dicts():
    dicts = [{}, {}]
    result = concat_dicts(dicts)
    assert result == {}


def test_concat_single_dict():
    dicts = [{"a": 1, "b": 2}]
    result = concat_dicts(dicts)
    assert result == {"a": [1], "b": [2]}


def test_concat_overlapping_keys_values():
    dicts = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"b": 5, "c": 6}
    ]
    result = concat_dicts(dicts)
    assert result == {"a": [1, 3], "b": [2, 4, 5], "c": [6]}
