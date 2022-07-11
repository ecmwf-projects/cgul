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
# Contents:
#   The standard coordinates implemented for CADS. This could become the
#   minimum standard coordinates for other projects.
import typing as T

COORDINATES: T.Dict[str, T.Any] = {
    "a_template": {
        "standard_name": "cf_standard_name",
        "units": "1",
        "axis": "TXYZ",
        "long_name": "A template coordinate",
        "comment": "Describe the coordinate to assist users.",
        "out_name": "a_template",
        "positive": "",
        "stored_direction": "",
        "type": "",
        "valid_max": "",
        "valid_min": "",
    },
    "altitude": {
        "standard_name": "altitude",
        "units": "m",
        "axis": "Z",
        "long_name": "Altitude",
        "positive": "up",
        "comment": (
            "Altitude is the (geometric) height above the geoid, which is the reference geopotential "
            "surface. The geoid is similar to mean sea level."
        ),
        "type": "real",
    },
    "dayofyear": {
        "standard_name": "",
        "units": "1",
        "axis": "T",
        "long_name": "Day of year",
        "positive": "up",
        "type": "integer",
        "valid_max": "366",
        "valid_min": "1",
    },
    "depth": {
        "standard_name": "depth",
        "units": "m",
        "axis": "Z",
        "long_name": "Depth",
        "comment": "Depth is the vertical distance below the surface",
        "positive": "down",
    },
    "forecast_reference_time": {
        "standard_name": "forecast_reference_time",
        # "units": "hours since ?",
        "axis": "T",
        "long_name": "Initial time of forecast",
        "positive": "up",
        "stored_direction": "increasing",
        "type": "double",
    },
    "height": {
        "standard_name": "height",
        "units": "m",
        "axis": "Z",
        "long_name": "height",
        "positive": "up",
        "stored_direction": "increasing",
        "type": "double",
    },
    "latitude": {
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
        "long_name": "latitude",
        "positive": "North",
        "type": "double",
        "valid_max": "90.0",
        "valid_min": "-90.0",
    },
    "leadtime": {
        "standard_name": "forecast_period",
        "units": "hours",
        "axis": "T",
        "long_name": "hours since forecast_reference_time",
        "positive": "up",
        "stored_direction": "increasing",
        "type": "double",
    },
    "leadtime_month": {
        "standard_name": "forecast_period",
        "units": "months",
        "axis": "T",
        "long_name": "months since forecast_reference_time",
        "positive": "up",
        "stored_direction": "increasing",
        "type": "double",
    },
    "longitude": {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
        "long_name": "longitude",
        "positive": "East",
        "type": "double",
        "valid_max": "360.0",
        "valid_min": "-180.0",
    },
    "plev": {
        "standard_name": "air_pressure",
        "units": "Pa",
        "axis": "Z",
        "long_name": "Pressure level",
        "positive": "down",
        "stored_direction": "decreasing",
        "type": "double",
    },
    "realization": {
        "standard_name": "realization",
        "units": "1",
        "axis": "",
        "long_name": "realization",
        "comment": (
            "Realization is used to label a dimension that can be thought of as a statistical sample, "
            "e.g., labelling members of a model ensemble."
        ),
        "positive": "",
        "stored_direction": "increasing",
        "type": "integer",
        "valid_max": "",
        "valid_min": "0",
    },
    "time": {
        "standard_name": "time",
        # "units": "hours since ?",
        "axis": "T",
        "long_name": "time",
        "positive": "up",
        "stored_direction": "increasing",
        "type": "double",
        "valid_max": "",
        "valid_min": "",
        "value": "",
        "z_bounds_factors": "",
        "z_factors": "",
        "bounds_values": "",
    },
    "year": {
        "standard_name": "",
        "units": "years",
        "axis": "T",
        "long_name": "year",
        "positive": "up",
        "stored_direction": "increasing",
        "type": "integer",
    },
}
