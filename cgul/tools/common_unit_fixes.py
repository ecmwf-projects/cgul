import typing as T

import xarray as xr

# Common units which are not recognised by cf-units.
# This is the default dictionary which will grow with experience
# and can be superceded/expanded when translate_coords/harmonise is called.
COMMON_UNIT_NAMES = {
    "-": "1",
    "DegNorth": "Degrees_North",
    "DegEast": "Degrees_East",
    "(0-1)": "1",
    "m of water equivalent": "m",
    "dimensionless": "1",
    "Dimensionless": "1",
    # The following are to ensure that xarray does not interpret
    # integer/float type data for datetime-objects. This is from
    # the cdscdm, and I do not know yet if it is necessary here.
    # "days": "day",
    # "seconds": "s",
    # "hours": "hour",
}


def common_unit_fixes(
    data: xr.DataArray,
    common_unit_names: T.Union[T.Dict[str, str], None] = None,
) -> xr.DataArray:
    """
    Apply common fixes to the units of xarray.DataArray objects.

    Parameters
    ----------
    data : xarray.DataArray

    Returns
    -------
    xarray.DataArray
    """
    if common_unit_names is None:
        common_unit_names = COMMON_UNIT_NAMES
    # Uncapitalise Units attribute
    if "Units" in data.attrs:
        data = data.assign_attrs({"units": data.attrs["Units"]})  # type: ignore
        data.attrs.pop("Units")

    # Common units that need renaming
    if data.attrs.get("units", "") in list(common_unit_names):
        data = data.assign_attrs({"units": common_unit_names[data.attrs["units"]]})  # type: ignore

    return data
