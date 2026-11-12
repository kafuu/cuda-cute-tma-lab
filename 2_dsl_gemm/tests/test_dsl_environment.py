import importlib.util

import pytest


def test_cutlass_dsl_is_available() -> None:
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("nvidia-cutlass-dsl is not installed")

    import cutlass  # noqa: F401


def test_layout_reference_does_not_need_gpu() -> None:
    from dsl_gemm.layout_reference import make_row_major_layout

    layout = make_row_major_layout((2, 2))
    assert layout.offset((1, 1)) == 3
