import logging

import cf_units
import xarray as xr

from .error_handler import error_handler


def convert_units(
    data: xr.DataArray,
    target_units: str,
    source_units: str,
    logger: logging.Logger = logging.getLogger(__name__),
    error_mode: str = "warn",
) -> xr.DataArray:
    """
    Convert units of an xarray.DataArray using cf-units relationships.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with units source_units.
    target_units : string
        Units to convert the data to.
    source_units : string
        Units to convert the data from.
    error_mode : str
        Error mode, options are "ignore": all conversion errors are ignored;
        "warn": conversion errors provide a stderr warning message; "raise":
        conversion errors raise a RuntimeError

    Returns
    -------
    xarray.DataArray
        Data array with units target_units
    """
    if target_units == source_units:
        return data

    try:
        _target_units = cf_units.Unit(target_units)
    except ValueError:
        error_handler(
            f"Target units for {data.name} ({target_units}) are not recognised by cf-units.\n",
            logger,
            warn_extra="Units will not be converted.\n",
            error_mode=error_mode,
        )
        return data

    try:
        _source_units = cf_units.Unit(source_units)
    except ValueError:
        error_handler(
            f"Source units for {data.name} ({source_units}) are not recognised by cf-units.\n",
            logger,
            warn_extra="Units will not be converted.\n",
            error_mode=error_mode,
        )
        return data

    try:
        # cf-units not compatible with xarray objects, so operate at the numpy level
        converted_values = _source_units.convert(data.values, _target_units)
    except Exception as err:
        error_handler(
            f"Error while converting {_source_units} to {_target_units} for {data.name}.\n",
            logger,
            warn_extra="Units will not be converted.\n",
            err=err,
            error_mode=error_mode,
        )
        return data

    # cf-units not compatible with xarray objects, so operate at the numpy level
    data = (data * 0) + converted_values
    data.assign_attrs({"units": source_units})  # type: ignore
    return data
