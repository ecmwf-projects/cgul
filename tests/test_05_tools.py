import _test_objects

import cgul

# Create test data array and dataset to apply methods to
TEST_DA = _test_objects.TEST_DA
TEST_DS = _test_objects.TEST_DS
RESULT_DA = _test_objects.RESULT_DA
RESULT_DS = _test_objects.RESULT_DS


def test_convert_units() -> None:
    result = cgul.tools.convert_units(
        TEST_DA["Depth"], source_units="km", target_units="m"
    )
    assert all(result.values == RESULT_DA["depth"].values)


def test_common_unit_fixes() -> None:
    result = cgul.tools.common_unit_fixes(
        TEST_DA.assign_attrs({"Units": "m of water equivalent"})  # type: ignore
    )
    assert result.attrs == TEST_DA.assign_attrs({"units": "m"}).attrs  # type: ignore

    result = cgul.tools.common_unit_fixes(TEST_DA.assign_attrs({"Units": "(0-1)"}))  # type: ignore
    assert result.attrs == TEST_DA.assign_attrs({"units": "1"}).attrs  # type: ignore

    result = cgul.tools.common_unit_fixes(
        TEST_DA.assign_attrs({"Units": "Dimensionless"})  # type: ignore
    )
    assert result.attrs == TEST_DA.assign_attrs({"units": "1"}).attrs  # type: ignore
