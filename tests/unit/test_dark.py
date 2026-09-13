from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import shuck.calibration.dark as dark_module
from shuck.calibration.dark import build_master_dark, write_master_dark


def test_build_master_dark_combines_and_records_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "a.fits": np.array([[1.0, 10.0]]),
        "b.fits": np.array([[1.0, 12.0]]),
        "c.fits": np.array([[1.0, 1000.0]]),
    }

    monkeypatch.setattr(dark_module, "read_ishell_raw", lambda path: Path(path))

    def process(path: Path, calibration: object) -> SimpleNamespace:
        del calibration
        return SimpleNamespace(
            image=values[path.name],
            variance=np.ones((1, 2)),
            mask=np.zeros((1, 2), dtype=np.uint8),
            metadata=SimpleNamespace(itime=10.0, ndr=2, coadds=1),
        )

    monkeypatch.setattr(dark_module, "process_raw_frame", process)

    result = build_master_dark(tuple(values), object())

    np.testing.assert_allclose(result.image, [[1.0, 11.0]])
    np.testing.assert_allclose(result.variance, [[1.0 / 3.0, 0.5]])
    assert result.rejected_fraction == pytest.approx(1.0 / 6.0)


def test_write_master_dark_preserves_arrays_and_input_provenance(tmp_path: Path) -> None:
    values = np.array([[1.0, 2.0]])
    product = dark_module.MasterDark(
        image=values,
        variance=values / 10,
        mask=np.array([[0, 1]], dtype=np.uint8),
        input_files=(Path("one.fits"), Path("two.fits")),
        sigma_clip=8.0,
        rejected_fraction=0.125,
    )

    output = write_master_dark(product, tmp_path / "master.fits", dark_id="D01")

    with fits.open(output) as hdus:
        np.testing.assert_allclose(hdus[0].data, values)
        np.testing.assert_allclose(hdus["VARIANCE"].data, values / 10)
        np.testing.assert_array_equal(hdus["MASK"].data, [[0, 1]])
        assert hdus[0].header["STAGE"] == "MASTER_DARK"
        assert hdus[0].header["DARKID"] == "D01"
        assert list(hdus[0].header["HISTORY"]) == ["INPUT one.fits", "INPUT two.fits"]
