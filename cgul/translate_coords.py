#
# Copyright 2017-2021 European Centre for Medium-Range Weather Forecasts (ECMWF).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
#   Edward Comyn-Platt - ECMWF - https://ecmwf.int
#   Alessandro Amici - B-Open - https://bopen.eu
#

import logging
import typing as T

import cf_units
import xarray as xr

from . import coordinate_models

LOG = logging.getLogger(__name__)

DEFAULT_COORD_MODEL = coordinate_models.CADS

# Common units which are not recognised by cf-units.
# This is the default dictionary which will grow with experience
# and can be superceded/expanded when translate_coords is called.
COMMON_UNIT_NAMES = {
    "-": "1",
    "DegNorth": "Degrees_North",
    "DegEast": "Degrees_East",
}


def error_handler(
    message: str,
    err: T.Union[Exception, str] = "",
    warn_extra: str = "",
    error_mode: str = "warn",
) -> None:
    if error_mode == "ignore":
        pass
    elif error_mode == "raise":
        if err:
            message = f"{message}\nTraceback:\n{err}"
        raise RuntimeError(message)
    else:
        if warn_extra:
            message = f"{message} {warn_extra}"
        if err:
            message = f"{message}\nTraceback:\n{err}"
        LOG.warning(message)


def convert_units(
    data: xr.DataArray,
    target_units: str,
    source_units: str,
    error_mode: str = "warn",
) -> xr.DataArray:
    """
    Convert units of the coordinate using cf-units relationships.

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
            warn_extra="Units will not be converted.\n",
            error_mode=error_mode,
        )
        return data

    try:
        _source_units = cf_units.Unit(source_units)
    except ValueError:
        error_handler(
            f"Source units for {data.name} ({source_units}) are not recognised by cf-units.\n",
            warn_extra="Units will not be converted.\n",
            error_mode=error_mode,
        )
        return data

    try:
        # cf-units not compatible with xarray objects, so operate at the numpy level
        converted_values = _source_units.convert(data.values, _target_units)
    except Exception as err:
        error_handler(
            f"Error while converting {_source_units} to {_target_units} for {data.name} coordinate.\n",
            warn_extra="Units will not be converted.\n",
            err=err,
            error_mode=error_mode,
        )
        return data

    # cf-units not compatible with xarray objects, so operate at the numpy level
    data = (data * 0) + converted_values
    data.assign_attrs({"units": source_units})  # type: ignore
    return data


def coord_translator(
    coord: xr.DataArray,
    c_model: T.Dict[str, T.Any],
    common_unit_names: T.Dict[str, str] = COMMON_UNIT_NAMES,
    error_mode: str = "warn",
) -> xr.DataArray:
    """
    Translate the coordinate based on the standard attributes/description.

    Parameters
    ----------
    coord : xarray.DataArray
        Coordinate dataarray to ranslate.
    c_model : dictionary
        A dictionary providing the attributes (including units) to transalte the input
        coordinate dataarray to.
    common_unit_names : dictionary
        A dictionary providing mapping of common names for units which are not recognised
        by cf-units to recognised cf-units, e.g. {'DegNorth': 'Degrees_North'}.
    error_mode : str
        Error mode, options are "ignore": all conversion errors are ignored;
        "warn": conversion errors provide a stderr warning message; "raise":
        conversion errors raise a RuntimeError

    Returns
    -------
    xarray.DataArray
        Data array for the coordinate translated to a format described by c_model
    """

    if "units" in coord.attrs:
        source_units = str(coord.attrs.get("units"))
        source_units = common_unit_names.get(source_units, source_units)
        target_units = c_model.get("units", source_units)
        coord = convert_units(
            coord,
            target_units,
            source_units,
            error_mode=error_mode,
        )

    coord_attrs = coord.attrs
    coord_attrs.update(c_model)
    coord.assign_attrs(coord_attrs)  # type: ignore

    return coord


def translate_coords(
    data: T.Union[xr.Dataset, xr.DataArray],
    coord_model: T.Dict[str, T.Any] = DEFAULT_COORD_MODEL,
    common_unit_names: T.Dict[str, str] = COMMON_UNIT_NAMES,
    error_mode: str = "warn",
) -> T.Union[xr.Dataset, xr.DataArray]:
    """
    Translate the coordiantes of an xarray dataset to a given coordinate model.

    Parameters
    ----------
    data : xarray.dataset
        Dataset with the coordinates to be translated.
    coord_model : dictionary
        A dictionary providing the coordinate model to transalte the input
        dataset to.
    common_unit_names : dictionary
        A dictionary providing mapping of common names for units which are not recognised
        by cf-units to recognised cf-units, e.g. {'DegNorth': 'Degrees_North'}.
    error_mode : str
        Error mode, options are "ignore": all conversion errors are ignored;
        "warn": conversion errors provide a stderr warning message; "raise":
        conversion errors raise a RuntimeError

    Returns
    -------
    xarray.Dataset
        Dataset with coordinates translated to those described by coord_model
    """
    for coordinate in data.coords:
        try:
            if coord_model.get("_always_lower_case", False):
                _coordinate = str(coordinate).lower()
            else:
                _coordinate = str(coordinate)
            c_model = coord_model.get(_coordinate, {})
            out_name = c_model.get("out_name", _coordinate)
            data = data.assign_coords(  # type: ignore
                {
                    coordinate: coord_translator(
                        data.coords[coordinate],
                        c_model,
                        common_unit_names=common_unit_names,
                        error_mode="warn",
                    )
                }
            )
            data = data.rename({coordinate: out_name})
        except Exception as err:
            if error_mode == "ignore":
                pass
            elif error_mode == "raise":
                raise RuntimeError(
                    f"Error while translating coordinate: {coordinate}.\n Traceback:\n{err}"
                )
            else:
                LOG.warning(
                    f"Error while translating coordinate: {coordinate}.\n Traceback:\n{err}"
                )

    return data
