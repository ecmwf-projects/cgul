import _test_objects
import xarray as xr

import cgul

# Create test data array and dataset to apply methods to
TEST_DA = _test_objects.TEST_DA
TEST_DS = _test_objects.TEST_DS
RESULT_DA = _test_objects.RESULT_DA
RESULT_DS = _test_objects.RESULT_DS


def test_translate_coords_dataset() -> None:
    result = cgul.translate_coords(TEST_DS, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DS, result)
    assert result.to_dict() == RESULT_DS.to_dict()  # type: ignore


def test_translate_coords_dataarray() -> None:
    result = cgul.translate_coords(TEST_DA, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DA, result)
    assert result.to_dict() == RESULT_DA.to_dict()


def test_coord_translator() -> None:
    result = cgul.coord_translator(
        TEST_DA["Lat"], c_model=cgul.coordinate_models.CADS["lat"]
    )
    result = result.assign_coords({"Lat": result})  # type: ignore
    RESULT = RESULT_DA["latitude"].rename({"latitude": "Lat"})
    RESULT.name = "Lat"
    xr.testing.assert_identical(RESULT, result)
    assert result.to_dict() == RESULT.to_dict()
