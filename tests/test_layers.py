from peagle_q.layers import resolve_layer_indices


def test_default_layer_selection_for_32_layers() -> None:
    assert resolve_layer_indices(32) == [4, 16, 28]


def test_explicit_layer_selection_is_sorted_and_unique() -> None:
    assert resolve_layer_indices(32, explicit=[16, 4, 16, 28]) == [4, 16, 28]
