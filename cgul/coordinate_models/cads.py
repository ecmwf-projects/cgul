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
#   Mapping of coordinate names to be applied as standard when
#   using the CADS coordiante model in cgul.
#
import typing as T

from .cads_coordinates import COORDINATES

for coord in COORDINATES:
    COORDINATES[coord].setdefault("out_name", coord)

COMMON_LAT_NAMES: T.List[str] = [
    "lat",
    "latitude",
]

COMMON_LON_NAMES: T.List[str] = [
    "lon",
    "long",
    "longitude",
]

CADS: T.Dict[str, T.Any] = {
    "_always_lower_case": True,
    **COORDINATES,
    **{lat: COORDINATES["latitude"] for lat in COMMON_LAT_NAMES},
    **{lon: COORDINATES["longitude"] for lon in COMMON_LON_NAMES},
    "valid_time": COORDINATES["time"],
    "step": COORDINATES["leadtime"],
    "forecastmonth": COORDINATES["leadtime_month"],
    "depthbelowland": COORDINATES["depth"],
    "isobaricinhpa": COORDINATES["plev"],
    "number": COORDINATES["realization"],
}
