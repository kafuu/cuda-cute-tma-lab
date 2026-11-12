from dsl_gemm.layout_reference import Layout, linear_size, make_row_major_layout


def test_row_major_matrix_layout() -> None:
    layout = make_row_major_layout((3, 4))

    assert layout.shape == (3, 4)
    assert layout.stride == (4, 1)
    assert layout.offset((1, 2)) == 6
    assert layout.offset((2, 3)) == 11
    assert linear_size(layout) == 12


def test_row_major_three_dimensional_layout() -> None:
    layout = make_row_major_layout((2, 3, 4))

    assert layout.stride == (12, 4, 1)
    assert layout.offset((1, 2, 3)) == 23


def test_explicit_layout() -> None:
    tiled = Layout(shape=(32, 33), stride=(33, 1))

    assert tiled.offset((1, 0)) == 33
