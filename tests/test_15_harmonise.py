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
    assert result.to_dict() == RESULT_DS.to_dict()  # type: ignore

    TEST_DS["test"] = TEST_DS["test"].assign_attrs({"Units": "m of water equivalent"})  # type: ignore
    result = cgul.harmonise(TEST_DS, coord_model=cgul.coordinate_models.CADS)
    RESULT_DS["test"] = RESULT_DS["test"].assign_attrs({"units": "m"})  # type: ignore
    xr.testing.assert_identical(RESULT_DS, result)
    assert result.to_dict() == RESULT_DS.to_dict()  # type: ignore


def test_harmonise_dataarray() -> None:
    result = cgul.harmonise(TEST_DA, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DA, result)
    assert result.to_dict() == RESULT_DA.to_dict()

    test_da = TEST_DA.assign_attrs({"Units": "m of water equivalent"})  # type: ignore
    result = cgul.harmonise(test_da, coord_model=cgul.coordinate_models.CADS)
    RESULT = RESULT_DA.assign_attrs({"units": "m"})  # type: ignore
    xr.testing.assert_identical(RESULT, result)
    assert result.to_dict() == RESULT.to_dict()
