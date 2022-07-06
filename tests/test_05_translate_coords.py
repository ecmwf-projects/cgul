import xarray as xr

import cgul

# Create test data array and dataset to apply methods to
TEST_DA = xr.DataArray(
    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    name="test",
    coords={
        "Depth": [0, 1],
        "Lat": [0, 1],
        "Lon": [0, 1],
    },
    dims=["Lat", "Lon", "Depth"],
)
TEST_DA["Depth"] = TEST_DA["Depth"].assign_attrs({"units": "km"})  # type: ignore
TEST_DA["Lat"] = TEST_DA["Lat"].assign_attrs({"units": "DegNorth"})  # type: ignore
TEST_DA["Lon"] = TEST_DA["Lon"].assign_attrs({"units": "Degrees_East"})  # type: ignore
TEST_DS = xr.Dataset({"test": TEST_DA})

# Create result data array and dataset to apply methods to
RESULT_DA = xr.DataArray(
    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    name="test",
    coords={
        "depth": [0, 1e3],
        "latitude": [0, 1],
        "longitude": [0, 1],
    },
    dims=["latitude", "longitude", "depth"],
)
RESULT_DA["depth"] = RESULT_DA["depth"].assign_attrs(cgul.coordinate_models.CADS["depth"])  # type: ignore
RESULT_DA["latitude"] = RESULT_DA["latitude"].assign_attrs(
    cgul.coordinate_models.CADS["latitude"]
)  # type: ignore
RESULT_DA["longitude"] = RESULT_DA["longitude"].assign_attrs(
    cgul.coordinate_models.CADS["longitude"]
)  # type: ignore
RESULT_DS = xr.Dataset({"test": RESULT_DA})


def test_translate_coords_dataset() -> None:
    result = cgul.translate_coords(TEST_DS, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DS, result)


def test_translate_coords_dataarray() -> None:
    result = cgul.translate_coords(TEST_DA, coord_model=cgul.coordinate_models.CADS)
    xr.testing.assert_identical(RESULT_DA, result)


def test_coord_translator() -> None:
    result = cgul.coord_translator(
        TEST_DA["Lat"], c_model=cgul.coordinate_models.CADS["lat"]
    )
    result = result.assign_coords({"Lat": result})  # type: ignore
    RESULT = RESULT_DA["latitude"].rename({"latitude": "Lat"})
    RESULT.name = "Lat"
    xr.testing.assert_identical(RESULT, result)


def test_convert_units() -> None:
    result = cgul.convert_units(TEST_DA["Depth"], source_units="km", target_units="m")
    assert all(result.values == RESULT_DA["depth"].values)
