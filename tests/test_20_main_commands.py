import tempfile

import _test_objects
import click.testing

from cgul import __main__

# Create test data array and dataset to apply methods to
TEST_DS = _test_objects.TEST_DS
RESULT_DS = _test_objects.RESULT_DS


def test_cfgrib_cli_to_netcdf() -> None:
    runner = click.testing.CliRunner()

    with tempfile.NamedTemporaryFile() as tmp:
        TEST_DS.to_netcdf(tmp.name)
        res = runner.invoke(__main__.cgul_cli, ["harmonise", "--check", tmp.name])
        assert res.exit_code == 0

    with tempfile.NamedTemporaryFile() as tmp:
        TEST_DS.to_netcdf(tmp.name)
        res = runner.invoke(__main__.cgul_cli, ["harmonise", tmp.name])
        assert res.exit_code == 0
        assert res.output == ""
