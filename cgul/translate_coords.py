#
# Copyright 2017-2022, European Union.
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
from copy import deepcopy

import xarray as xr

from . import coordinate_models, tools

LOG = logging.getLogger(__name__)

DEFAULT_COORD_MODEL = coordinate_models.CADS


def coord_translator(
    coord: xr.DataArray,
    c_model: T.Dict[str, T.Any],
    common_unit_names: T.Union[T.Dict[str, str], None] = None,
    convert_units: bool = True,
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
    convert_units: bool
        A boolean flag to convert units to those defined in the coordinate model
    error_mode : str
        Error mode, options are "ignore": all conversion errors are ignored;
        "warn": conversion errors provide a stderr warning message; "raise":
        conversion errors raise a RuntimeError

    Returns
    -------
    xarray.DataArray
        Data array for the coordinate translated to a format described by c_model
    """
    coord = tools.common_unit_fixes(coord, common_unit_names=common_unit_names)
    if convert_units and ("units" in coord.attrs):
        source_units = str(coord.attrs.get("units"))
        target_units = c_model.get("units", source_units)
        coord = tools.convert_units(
            coord,
            target_units,
            source_units,
            LOG,
            error_mode=error_mode,
        )

    # Attributes in source data are given priority
    coord_attrs = {
        **c_model,
        **coord.attrs,
    }

    # Sometimes attributes are stored in the encoding (e.g. for time variables),
    # to remove conflicts when saving as netCDF we remove the attribute value here
    coord_attrs = {
        key: val for key, val in coord_attrs.items() if key not in coord.encoding.keys()
    }
    encoding_attrs = {
        key: val for key, val in coord_attrs.items() if key in coord.encoding.keys()
    }

    coord = coord.assign_attrs(coord_attrs)
    coord.encoding.update(encoding_attrs)

    return coord


def translate_coords(
    data: T.Union[xr.Dataset, xr.DataArray],
    coord_model: T.Union[T.Dict[str, T.Any], None] = None,
    common_unit_names: T.Union[T.Dict[str, str], None] = None,
    convert_units: bool = True,
    error_mode: str = "warn",
) -> T.Union[xr.Dataset, xr.DataArray]:
    """
    Translate the coordinates of an xarray dataset to a given coordinate model.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset with the coordinates to be translated.
    coord_model : dictionary
        A dictionary providing the coordinate model to transalte the input
        dataset to.
    common_unit_names : dictionary
        A dictionary providing mapping of common names for units which are not recognised
        by cf-units to recognised cf-units, e.g. {'DegNorth': 'Degrees_North'}. Default is
        cgul.tools.COMMON_UNIT_NAMES
    convert_units: bool
        A boolean flag to convert units to those defined in the coordinate model
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

    lower_case = coord_model.get("_always_lower_case", False)
    # First build dictionary of applicable coord models so we can deduce the order to change
    update_order = []
    c_models = {}
    for coordinate in data.coords:
        if lower_case:
            _coordinate = str(coordinate).lower()
        else:
            _coordinate = str(coordinate)
        # Prioritise standard_name in attributes (this fixes disagreement between grib and CF time vars)
        _coordinate_standard_name = data[coordinate].attrs.get(
            "standard_name", _coordinate
        )
        if _coordinate_standard_name in coord_model:
            _coordinate = _coordinate_standard_name
        c_model = coord_model.get(_coordinate, {})
        out_name = c_model.get("out_name", _coordinate)
        c_models[coordinate] = deepcopy(c_model)
        # Create safe order in which to update coordinates
        if out_name in update_order:
            update_order.append(coordinate)
        else:
            update_order.insert(0, coordinate)

    for coordinate in data.coords:
        c_model = c_models[coordinate]
        out_name = c_model.get("out_name", coordinate)
        try:
            data = data.assign_coords(
                {
                    coordinate: coord_translator(
                        data.coords[coordinate],
                        c_model,
                        common_unit_names=common_unit_names,
                        convert_units=convert_units,
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
