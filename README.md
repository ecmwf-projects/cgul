# cgul

**C**ommon **G**eneralised **U**niform and **L**ight. Python package which will be used to ensure consistent data format when working with Xarray type data objects.

## Usage

The primary function in cgul is `cgul.translate_coords`. This function operates on an `xarray.Dataset` or
`xarray.DataArray` and translates the coordinates to the specified coordinate model. Several coordinate models
are included in the package, and users are recommended to use the default CADS coordinate model. A typical
use case woould be:

```
import xarray as xr
import cgul

infile = 'data_file.nc'
data = xr.open_dataset(infile)

# To harmonise the coordinates and unit names:
data_cgul = cgul.harmonise(
    data,
    coord_model=cgul.coordinate_models.CADS  # This is the default value, so optional in this case
)

# To just harmonise the coordinates:
data_cgul = cgul.translate_coordinates(
    data,
    coord_model=cgul.coordinate_models.CADS  # This is the default value, so optional in this case
)
```

It is is also possible to use command line executables to check files can be harmonised, or to produce netCDF files with harmonised coordinates and metadata:

```
# To check that $INFILE has contents that can be harmonised,
# this will print out the harmonised xarray.Dataset:
cgul harmonise --check $INFILE

# To produce an ouput file, $OUTFILE, which contains the harmonised version of
# the contents of $INFILE:
cgul harmonise --output $OUTFILE $INFILE
```

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.10:

```
conda create -n DEVELOP -c conda-forge python=3.10
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Sync with the latest [template](https://github.com/ecmwf-projects/cookiecutter-conda-package) (optional): `make template-update`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Build the documentation (see [Sphinx tutorial](https://www.sphinx-doc.org/en/master/tutorial/)): `make docs-build`

## License

```
Copyright 2017-2022, European Union.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
