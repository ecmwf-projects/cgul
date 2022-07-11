import _test_objects
import xarray as xr

import cgul

# Create test data array and dataset to apply methods to
TEST_DA = _test_objects.TEST_DA
TEST_DS = _test_objects.TEST_DS
RESULT_DA = _test_objects.RESULT_DA
RESULT_DS = _test_objects.RESULT_DS


def test_harmonise_dataset() -> None:
    result = cgul.harmonise(TEST_DS, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DS, result)
    assert result["test"].attrs == RESULT_DS["test"].attrs

    TEST_DS["test"] = TEST_DS["test"].assign_attrs({"Units": "m of water equivalent"})
    result = cgul.harmonise(TEST_DS, coord_model=cgul.coordinate_models.CADS)
    RESULT_DS["test"] = RESULT_DS["test"].assign_attrs({"units": "m"})
    xr.testing.assert_identical(RESULT_DS, result)
    assert result["test"].attrs == RESULT_DS["test"].attrs


def test_harmonise_dataarray() -> None:
    result = cgul.harmonise(TEST_DA, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DA, result)
    assert result.attrs == RESULT_DA.attrs

    test_da = TEST_DA.assign_attrs({"Units": "m of water equivalent"})
    result = cgul.harmonise(test_da, coord_model=cgul.coordinate_models.CADS)
    result_da = RESULT_DA.assign_attrs({"units": "m"})
    xr.testing.assert_identical(result_da, result)
    assert result.attrs == result_da.attrs
