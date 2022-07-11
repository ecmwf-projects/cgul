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
RESULT_DA["depth"] = RESULT_DA["depth"].assign_attrs(
    cgul.coordinate_models.CADS["depth"]
)  # type: ignore
RESULT_DA["latitude"] = RESULT_DA["latitude"].assign_attrs(
    cgul.coordinate_models.CADS["latitude"]
)  # type: ignore
RESULT_DA["longitude"] = RESULT_DA["longitude"].assign_attrs(
    cgul.coordinate_models.CADS["longitude"]
)  # type: ignore
RESULT_DS = xr.Dataset({"test": RESULT_DA})
