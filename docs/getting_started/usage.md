.. \_overview:

# Usage

The primary function in cgul is `cgul.translate_coords`. This function operates on an `xarray.Dataset` or
`xarray.DataArray` and translates the coordinates to the specified coordinate model. Several coordinate models
are included in the package, and users are recommended to use the default CADS coordinate model. A typical
use case would be:

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
