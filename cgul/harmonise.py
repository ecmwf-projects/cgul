#
# Copyright 2017-2022, European Centre for Medium-Range Weather Forecasts (ECMWF).
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
#
import logging
import typing as T

import xarray as xr

from . import coordinate_models, tools, translate_coords

LOG = logging.getLogger(__name__)

DEFAULT_COORD_MODEL = coordinate_models.CADS


def harmonise(
    data: T.Union[xr.Dataset, xr.DataArray],
    coord_model: T.Union[T.Dict[str, T.Any], None] = None,
    additional_coords: T.List[str] = [],
    common_unit_names: T.Union[T.Dict[str, str], None] = None,
    convert_units: bool = True,
    error_mode: str = "warn",
) -> T.Union[xr.Dataset, xr.DataArray]:
    """
    Harmonise input xarray object to a coordinate model and adjust any common issues such as units names.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Dataset with the coordinates to be translated.
    coord_model : dictionary
        A dictionary providing the coordinate model to transalte the input
        dataset to.
    additional_coords : list
        A dictionary providing the coordinate model to transalte the input
        dataset to.
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
    xarray.Dataset
        Dataset with coordinates translated to those described by coord_model
    """
    if coord_model is None:
        coord_model = DEFAULT_COORD_MODEL

    # 1. assign any coordinate variables, which are not already assigned as
    #    coordinates.
    assign_coords = {
        coord: data[coord] for coord in additional_coords if coord in data.data_vars
    }
    data = data.assign_coords(assign_coords)  # type: ignore

    # 2. Apply common fixes to the attributes of data variables
    #    (coordinate variables are handled in translate_coords)
    if isinstance(data, xr.Dataset):
        for var in data.data_vars:
            data[var] = tools.common_unit_fixes(
                data[var], common_unit_names=common_unit_names
            )
    elif isinstance(data, xr.DataArray):
        data = tools.common_unit_fixes(data, common_unit_names=common_unit_names)

    # 3. Translate coordinates to the chosen coordinate model
    data = translate_coords.translate_coords(
        data,
        coord_model=coord_model,
        common_unit_names=common_unit_names,
        convert_units=convert_units,
        error_mode=error_mode,
    )

    return data
