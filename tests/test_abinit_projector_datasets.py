"""Story 061 tests: UPF/PAW-XML projector/radial dataset readers.

Real pinned fixtures only (Architecture §9 SHA256 verified); no mock dataset
objects.  Fails-closed diagnostics D1-D4 are demonstrated on minimally
corrupted copies written into tmp_path at test time.
"""

import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_LAWAF_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _LAWAF_PKG_ROOT not in sys.path:
    sys.path.insert(0, _LAWAF_PKG_ROOT)

from lawaf.interfaces.abinit.basis_datasets import (
    DatasetError,
    PawXmlDataset,
    UpfProjectorDataset,
    read_paw_xml_dataset,
    read_upf_projectors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NC_DIR = REPO_ROOT / "abinit/tests/Pspdir/Psdj_nc_fr_04_pbe_std_psp8"
PAW_DIR = REPO_ROOT / "abinit/tests/Pspdir/Psdj_paw_pbe_std"

PINNED_SHA256 = {
    "Ga.upf": "e1aad2af1f8e69635634fd601d945943e033a36a1bb404cd4a1134e67180de7d",
    "As.upf": "74e21823d94edd4762de345811a3ce82f282f2768debef657399a61cfae3c7a2",
    "Ga.xml": "10b9b72c90d5f758653efd89f5871bb43cf4e1e354e7683b8cbf65619e771ba4",
    "As.xml": "e04d9bb679cd9e63cf8ba1303bc285e785727cc83664d57a847422556ad06b40",
}


@pytest.fixture(scope="module", params=["Ga.upf", "As.upf"])
def upf_path(request):
    path = NC_DIR / request.param
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_SHA256[path.name]
    return path


@pytest.fixture(scope="module", params=["Ga.xml", "As.xml"])
def paw_path(request):
    path = PAW_DIR / request.param
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_SHA256[path.name]
    return path


def _corrupted_upf(tmp_path, source: Path, transform) -> Path:
    """Copy a pinned UPF and apply an ET-tree transform, writing to tmp_path."""
    tree = ET.parse(source)
    transform(tree.getroot())
    out = tmp_path / source.name
    tree.write(out, encoding="unicode", xml_declaration=False)
    return out


# ---------------------------------------------------------------------------
# TEST-061-01: pinned UPF pair channel/label/grid assertions (AC-061-01)
# ---------------------------------------------------------------------------


class TestUpfChannels:
    expected = {
        "Ga.upf": (
            ["3D", "3D", "4S", "4P", "4P"],
            [2, 2, 0, 1, 1],
            1926,
        ),
        "As.upf": (
            ["3D", "3D", "4S", "4P", "4P"],
            [2, 2, 0, 1, 1],
            1358,
        ),
    }

    def test_061_01_channels_labels_l(self, upf_path):
        ds = read_upf_projectors(upf_path)
        labels, ls, nmesh = self.expected[upf_path.name]
        assert isinstance(ds, UpfProjectorDataset)
        assert [c.label for c in ds.channels] == labels
        assert [c.l for c in ds.channels] == ls
        assert [c.radial_id for c in ds.channels] == [
            f"PP_CHI.{i}" for i in range(1, len(labels) + 1)
        ]
        # Ga channel 3 resolves as 4S,l0 and channel 4 as 4P,l1
        if upf_path.name == "Ga.upf":
            assert (ds.channels[2].label, ds.channels[2].l) == ("4S", 0)
            assert (ds.channels[3].label, ds.channels[3].l) == ("4P", 1)

    def test_061_01_grid_as_stored(self, upf_path):
        ds = read_upf_projectors(upf_path)
        nmesh = self.expected[upf_path.name][2]
        # grid + quadrature recorded as stored: no resampling, no double r,
        # no R/rR relabeling
        assert ds.grid.representation == "r"
        assert ds.grid.quadrature_weights is not None
        assert len(ds.grid.r) == len(ds.grid.quadrature_weights) == nmesh
        assert np.all(np.diff(ds.grid.r) > 0)
        assert ds.grid.r[0] == 0.0
        # values are exactly as stored, one array per channel
        for ch in ds.channels:
            assert ch.values.shape == (nmesh,)
            assert ch.grid is ds.grid
        # spot-check the array content against the raw file text
        raw = ET.parse(upf_path).getroot()
        r_text = raw.find("./PP_MESH/PP_R").text.split()
        assert ds.grid.r[1] == float(r_text[1])
        chi3 = next(raw.iter("PP_CHI.3")).text.split()
        assert ds.channels[2].values[0] == float(chi3[0])
        assert ds.channels[2].values[-1] == float(chi3[-1])

    def test_061_01_channel_lookup(self, upf_path):
        ds = read_upf_projectors(upf_path)
        ch = ds.get_channel("PP_CHI.3")
        assert ch.label == "4S" and ch.l == 0


# ---------------------------------------------------------------------------
# TEST-061-02: pinned XML pair six-channel structure (AC-061-02)
# ---------------------------------------------------------------------------


class TestPawXmlChannels:
    def test_061_02_six_channels(self, paw_path):
        ds = read_paw_xml_dataset(paw_path)
        assert isinstance(ds, PawXmlDataset)
        assert [(c.n, c.l) for c in ds.channels] == [
            (4, 0),
            (0, 0),
            (4, 1),
            (0, 1),
            (3, 2),
            (0, 2),
        ]
        n = len(ds.grid.r)
        assert n == 1500
        assert np.all(np.diff(ds.grid.r) > 0)
        for ch in ds.channels:
            assert ch.ae_partial_wave.shape == (n,)
            assert ch.pseudo_partial_wave.shape == (n,)
            assert ch.dual_projector.shape == (n,)

    def test_xml_radial_measure_preserves_projector_duality(self, paw_path):
        from scipy.integrate import simpson

        ds = read_paw_xml_dataset(paw_path)
        r = ds.grid.r
        # Interpret the public representation exactly as a contraction consumer does.
        measure = r * r if ds.radial_representation == "R" else np.ones_like(r)
        jacobian = ds.grid.d * (r + ds.grid.a)
        for angular_momentum in sorted({c.l for c in ds.channels}):
            block = [c for c in ds.channels if c.l == angular_momentum]
            overlap = np.array(
                [
                    [
                        simpson(
                            p.dual_projector
                            * t.pseudo_partial_wave
                            * measure
                            * jacobian,
                            dx=1.0,
                        )
                        for t in block
                    ]
                    for p in block
                ]
            )
            # Physical duality on the declared log mesh; frozen PAW metric target.
            np.testing.assert_allclose(overlap, np.eye(len(block)), rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------
# TEST-061-03: j partners from PP_SPIN_ORB/PP_RELWFC + occupation cross-check
# ---------------------------------------------------------------------------

EXPECTED_J = {
    # per PP_CHI index: (lchi, jchi, nn) parsed from PP_RELWFC
    "Ga.upf": [(2, 2.5, 1), (2, 1.5, 1), (0, 0.5, 2), (1, 1.5, 3), (1, 0.5, 3)],
    "As.upf": [(2, 2.5, 1), (2, 1.5, 1), (0, 0.5, 2), (1, 1.5, 3), (1, 0.5, 3)],
}


class TestSpinOrbitMetadata:
    def test_061_03_j_partners_parsed(self, upf_path):
        ds = read_upf_projectors(upf_path)
        partners = ds.spin_orbit_partners()
        assert [(p.lchi, p.jchi, p.nn) for p in partners] == EXPECTED_J[upf_path.name]
        assert [p.index for p in partners] == [1, 2, 3, 4, 5]

    def test_061_03_occupation_crosscheck_passes(self, upf_path):
        ds = read_upf_projectors(upf_path)
        # asserted cross-check only: must pass on all ten pinned channels and
        # must never be used to derive j
        ds.assert_occupation_consistency()

    def test_061_03_missing_j_block_fails_closed(self, upf_path, tmp_path):
        def strip_spin_orb(root):
            for el in root.findall("PP_SPIN_ORB"):
                root.remove(el)

        corrupted = _corrupted_upf(tmp_path, upf_path, strip_spin_orb)
        ds = read_upf_projectors(corrupted)
        with pytest.raises(DatasetError) as exc:
            ds.spin_orbit_partners()
        msg = str(exc.value)
        assert "PP_SPIN_ORB" in msg and "PP_RELWFC" in msg
        assert upf_path is not None and "spin" in msg.lower()


# ---------------------------------------------------------------------------
# TEST-061-05 (D1-D4): fail-closed diagnostics on corrupted copies
# ---------------------------------------------------------------------------


class TestFailClosedD1toD4:
    def test_d1_unknown_upf_version(self, upf_path, tmp_path):
        def bump_version(root):
            root.set("version", "9.9")

        corrupted = _corrupted_upf(tmp_path, upf_path, bump_version)
        with pytest.raises(DatasetError) as exc:
            read_upf_projectors(corrupted)
        msg = str(exc.value)
        assert "9.9" in msg and "2.0.1" in msg  # declared vs supported list

    def test_d2_requested_radial_id_absent(self, upf_path):
        ds = read_upf_projectors(upf_path)
        with pytest.raises(DatasetError) as exc:
            ds.get_channel("PP_CHI.6")
        msg = str(exc.value)
        assert "PP_CHI.6" in msg
        assert "PP_CHI.1" in msg and "PP_CHI.5" in msg  # available IDs

    def test_d3_channel_length_mismatch(self, upf_path, tmp_path):
        def truncate_chi3(root):
            el = next(root.iter("PP_CHI.3"))
            el.set("size", "42")

        corrupted = _corrupted_upf(tmp_path, upf_path, truncate_chi3)
        mesh_size = TestUpfChannels.expected[upf_path.name][2]
        with pytest.raises(DatasetError) as exc:
            read_upf_projectors(corrupted)
        msg = str(exc.value)
        assert "PP_CHI.3" in msg and str(mesh_size) in msg and "42" in msg

    def test_d3_paw_length_mismatch(self, paw_path, tmp_path):
        tree = ET.parse(paw_path)
        root = tree.getroot()
        ae = root.findall("ae_partial_wave")[2]
        ae.text = " ".join(ae.text.split()[:37])
        out = tmp_path / paw_path.name
        tree.write(out, encoding="unicode", xml_declaration=False)
        with pytest.raises(DatasetError) as exc:
            read_paw_xml_dataset(out)
        msg = str(exc.value)
        assert "1500" in msg and "37" in msg

    def test_d3_paw_non_monotonic_grid(self, paw_path, tmp_path):
        tree = ET.parse(paw_path)
        root = tree.getroot()
        values = root.find("./radial_grid/values")
        tokens = values.text.split()
        tokens[10], tokens[11] = tokens[11], tokens[10]
        values.text = " ".join(tokens)
        out = tmp_path / paw_path.name
        tree.write(out, encoding="unicode", xml_declaration=False)
        with pytest.raises(DatasetError) as exc:
            read_paw_xml_dataset(out)
        assert "monoton" in str(exc.value).lower()

    def test_d4_reported_above(self):
        # D4 is demonstrated by TestSpinOrbitMetadata.
        # test_061_03_missing_j_block_fails_closed
        assert True


# ---------------------------------------------------------------------------
# TEST-061-06: import graph contains no pypao/abinao (AC-061-06)
# ---------------------------------------------------------------------------


class TestImportGraph:
    def test_061_06_no_pypao_abinao_import(self):
        import ast

        pkg = Path(__file__).resolve().parents[1] / "lawaf" / "interfaces" / "abinit"
        for src in sorted(pkg.glob("*.py")):
            tree = ast.parse(src.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                else:
                    continue
                for mod in mods:
                    root = mod.split(".")[0]
                    assert root not in ("pypao", "abinao"), f"{src.name} imports {mod}"
        import lawaf.interfaces.abinit.basis_datasets  # noqa: F401
        import lawaf.interfaces.abinit.trial_manifest  # noqa: F401

        for token in ("pypao", "abinao"):
            assert token not in sys.modules


@pytest.mark.parametrize("kind", ["upf", "paw"])
def test_dataset_radial_arrays_cannot_change_shared_operator_data(kind):
    dataset = (
        read_upf_projectors(NC_DIR / "Ga.upf")
        if kind == "upf"
        else read_paw_xml_dataset(PAW_DIR / "Ga.xml")
    )
    arrays = [dataset.grid.r]
    if kind == "upf":
        arrays.append(dataset.grid.quadrature_weights)
        arrays.extend(channel.values for channel in dataset.channels)
    else:
        arrays.extend(
            array
            for channel in dataset.channels
            for array in (
                channel.ae_partial_wave,
                channel.pseudo_partial_wave,
                channel.dual_projector,
            )
        )
    for array in arrays:
        original = array.copy()
        with pytest.raises(ValueError):
            array[0] = 123
        np.testing.assert_array_equal(array, original)
