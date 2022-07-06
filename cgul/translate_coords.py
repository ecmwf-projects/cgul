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

import xarray as xr

from . import coordinate_models
from .tools import convert_units

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
            LOG,
            error_mode=error_mode,
        )

    coord_attrs = coord.attrs
    coord_attrs.update(c_model)
    coord.assign_attrs(coord_attrs)  # type: ignore

    return coord


def translate_coords(
    data: T.Union[xr.Dataset, xr.DataArray],
    coord_model: T.Union[T.Dict[str, T.Any], None] = None,
    common_unit_names: T.Union[T.Dict[str, str], None] = None,
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
    if coord_model is None:
        coord_model = DEFAULT_COORD_MODEL
    if common_unit_names is None:
        common_unit_names = COMMON_UNIT_NAMES

    for coordinate in data.coords:
        # try:
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
    # except Exception as err:
    #     if error_mode == "ignore":
    #         pass
    #     elif error_mode == "raise":
    #         raise RuntimeError(
    #             f"Error while translating coordinate: {coordinate}.\n Traceback:\n{err}"
    #         )
    #     else:
    #         LOG.warning(
    #             f"Error while translating coordinate: {coordinate}.\n Traceback:\n{err}"
    #         )

    return data
