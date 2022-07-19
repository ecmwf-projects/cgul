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

import json
import os.path
import typing as T

import click  # type: ignore

# NOTE: imports are executed inside functions so missing dependencies don't break all commands


def handle_json(in_json: str) -> T.Any:
    """
    Handle input json which can be a a json format string, or path to a json format file.

    Returns
    -------
    A dictionary of the json contents.
    """
    try:
        # Assume a json format string
        out_json = json.loads(in_json)
    except json.JSONDecodeError:
        # Then a json file
        with open(in_json, "r") as f:
            out_json = json.load(f)

    return out_json


@click.group()  # type: ignore
def cgul_cli() -> None:
    pass


@cgul_cli.command("harmonise")  # type: ignore
@click.argument("inpaths", nargs=-1)  # type: ignore
@click.option(
    "--check",
    "-C",
    is_flag=True,
    help="Option to check contents only and not produce output netCDF file.",
)  # type: ignore
@click.option(
    "--outpath", "-o", default=None, help="Filename of the output netcdf file."
)  # type: ignore
@click.option(
    "--coord-model",
    "-c",
    default="CADS",
    help=(
        "Coordinate model to translate the grib coordinates to. "
        "Can be a recoginised cgul coordinate model (CADS, CDS, ECMWF), "
        "or a JSON string or path to a JSON file with the full coordinate model."
    ),
)  # type: ignore
@click.option(
    "--additional-coords",
    "-a",
    default=None,
    help=(
        "List of data variables to be considered coordiante variables by xarray. "
        "Only required if the file metadata does not already allocate the variable[s] "
        "as coordiante[s]."
        "Can be provided as a comma space list, e.g. `-a lat,lon,depth`."
        "They can also be included in the harmonise-kwargs-json."
    ),
)  # type: ignore
@click.option(
    "--common-unit-names-json",
    "-c",
    default=None,
    help=(
        "A dictionary matching unit names in source data files to unit names recognised by cf-units. "
        "Can be provided as a JSON string or path to a JSON file."
    ),
)  # type: ignore
@click.option(
    "--harmonise-kwargs-json",
    "-h",
    default=None,
    help=(
        "kwargs used in cgul.harmonise."
        "Can either be a JSON format string or the path to JSON file."
        "components passed in via --coord-model, --additional-coords and --common-unit-names will "
        "supercede the information provided here."
    ),
)  # type: ignore
@click.option(
    "--xarray-open-kwargs-json",
    "-x",
    default=None,
    help=(
        "kwargs used in xarray.open_dataset."
        "Can either be a JSON format string or the path to JSON file."
    ),
)  # type: ignore
@click.option(
    "--netcdf-kwargs-json",
    "-n",
    default=None,
    help=(
        "kwargs used in xarray.to_netcdf when creating output netCDF file."
        "Can either be a JSON format string or the path to JSON file."
    ),
)  # type: ignore
def harmonise(
    inpaths: T.List[str],
    outpath: str,
    coord_model: str,
    additional_coords: str,
    common_unit_names_json: str,
    harmonise_kwargs_json: str,
    xarray_open_kwargs_json: str,
    netcdf_kwargs_json: str,
    check: bool,
) -> None:
    import xarray as xr

    import cgul

    if len(inpaths) == 0:
        return

    if xarray_open_kwargs_json is not None:
        xarray_open_kwargs = handle_json(xarray_open_kwargs_json)
    else:
        xarray_open_kwargs = {}

    xarray_open_kwargs.setdefault("decode_coords", "all")

    if len(inpaths) == 1:
        # avoid to depend on dask when passing only one file
        ds = xr.open_dataset(
            inpaths[0],
            **xarray_open_kwargs,
        )  # type: ignore
    else:
        xarray_open_kwargs.setdefault("combine", "by_coords")
        ds = xr.open_mfdataset(inpaths, **xarray_open_kwargs)  # type: ignore

    # Set up kwargs for cgul.harmonise
    harmonise_kwargs = {}
    if harmonise_kwargs_json is not None:
        harmonise_kwargs = handle_json(harmonise_kwargs_json)
    else:
        harmonise_kwargs = {}

    if coord_model is not None:
        try:
            coord_model_dict = getattr(cgul.coordinate_models, coord_model)
        except AttributeError:
            coord_model_dict = handle_json(coord_model)
        harmonise_kwargs.update({"coord_model": coord_model_dict})

    if additional_coords is not None:
        harmonise_kwargs.update({"additional_coords": additional_coords})

    if common_unit_names_json is not None:
        harmonise_kwargs.update(
            {"common_unit_names": handle_json(common_unit_names_json)}
        )

    # Call harmonise
    ds = cgul.harmonise(ds, **harmonise_kwargs)

    if check:
        # If only checking contents, print the harmonised xarray.Dataset
        print(ds)
    else:
        # Else, produce output netcdf file
        if not outpath:
            outpath = os.path.splitext(inpaths[0])[0] + "-cgul_harmony.nc"

        if netcdf_kwargs_json is not None:
            netcdf_kwargs = handle_json(netcdf_kwargs_json)
        else:
            netcdf_kwargs = {}

        ds.to_netcdf(outpath, **netcdf_kwargs)


if __name__ == "__main__":
    cgul_cli()
