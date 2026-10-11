"""Story 061 tests: trial-basis manifest schema and parser.

The default NC manifest selects radial IDs PP_CHI.3(4S,l0) and PP_CHI.4(4P,l1)
per species (Architecture §4.1/§9).  Diagnostics D5-D8 are demonstrated on
minimally corrupted manifest copies written into tmp_path at test time.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_LAWAF_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _LAWAF_PKG_ROOT not in sys.path:
    sys.path.insert(0, _LAWAF_PKG_ROOT)

from lawaf.interfaces.abinit.basis_datasets import DatasetError, read_upf_projectors
from lawaf.interfaces.abinit.trial_manifest import (
    TrialManifest,
    TrialRecord,
    default_nc_manifest,
    parse_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NC_DIR = REPO_ROOT / "abinit/tests/Pspdir/Psdj_nc_fr_04_pbe_std_psp8"
GA_UPF = NC_DIR / "Ga.upf"
AS_UPF = NC_DIR / "As.upf"

GA_SHA = "e1aad2af1f8e69635634fd601d945943e033a36a1bb404cd4a1134e67180de7d"
AS_SHA = "74e21823d94edd4762de345811a3ce82f282f2768debef657399a61cfae3c7a2"


def _write_default(tmp_path, **overrides):
    path = tmp_path / "basis_manifest.json"
    default_nc_manifest(
        path,
        species_datasets=[("Ga", GA_UPF), ("As", AS_UPF)],
        sites=[
            {"species": "Ga", "position": [0.0, 0.0, 0.0]},
            {"species": "As", "position": [0.25, 0.25, 0.25]},
        ],
        **overrides,
    )
    return path


# ---------------------------------------------------------------------------
# TEST-061-04: default manifest -> exactly 8/16 ordered trial records
# ---------------------------------------------------------------------------


class TestDefaultManifest:
    def test_061_04_eight_scalar_trials_in_order(self, tmp_path):
        path = _write_default(tmp_path)
        manifest = parse_manifest(path)
        assert isinstance(manifest, TrialManifest)
        records = manifest.trial_records
        assert len(records) == 8
        assert all(isinstance(r, TrialRecord) for r in records)
        # atom order -> radial channel -> harmonic index
        expected = [
            (0, "Ga", "PP_CHI.3", "4S", 0, 0, None),
            (0, "Ga", "PP_CHI.4", "4P", 1, 0, None),
            (0, "Ga", "PP_CHI.4", "4P", 1, 1, None),
            (0, "Ga", "PP_CHI.4", "4P", 1, 2, None),
            (1, "As", "PP_CHI.3", "4S", 0, 0, None),
            (1, "As", "PP_CHI.4", "4P", 1, 0, None),
            (1, "As", "PP_CHI.4", "4P", 1, 1, None),
            (1, "As", "PP_CHI.4", "4P", 1, 2, None),
        ]
        got = [
            (
                r.site_index,
                r.species,
                r.radial_id,
                r.radial_label,
                r.l,
                r.harmonic_index,
                r.spin_label,
            )
            for r in records
        ]
        assert got == expected

    def test_061_04_spin_frame_labels_double_count(self, tmp_path):
        path = _write_default(tmp_path, spin_frame_labels=True)
        records = parse_manifest(path).trial_records
        assert len(records) == 16
        assert {r.spin_label for r in records} == {"up", "down"}
        # spin label is the last ordering key within each spatial trial
        first_spatial = [
            (r.site_index, r.radial_id, r.harmonic_index, r.spin_label)
            for r in records[:2]
        ]
        assert first_spatial == [(0, "PP_CHI.3", 0, "up"), (0, "PP_CHI.3", 0, "down")]

    def test_061_04_records_carry_labels_and_metadata(self, tmp_path):
        path = _write_default(tmp_path)
        manifest = parse_manifest(path)
        assert manifest.harmonic_convention["name"]
        assert manifest.harmonic_convention["normalization"]
        assert len(manifest.harmonic_convention["local_axes"][0]) == 3
        assert manifest.units["length"] == "bohr"
        assert manifest.trial_definition == "pseudo_radial"
        assert manifest.species_fingerprints["Ga"] == GA_SHA
        assert manifest.species_fingerprints["As"] == AS_SHA
        rec = manifest.trial_records[0]
        assert rec.species == "Ga" and rec.radial_id == "PP_CHI.3"
        assert rec.harmonic_label == "s"

    def test_061_04_records_resolve_against_dataset(self, tmp_path):
        path = _write_default(tmp_path)
        manifest = parse_manifest(path)
        ds = read_upf_projectors(GA_UPF)
        rec = manifest.trial_records[0]
        ch = ds.get_channel(rec.radial_id)
        assert (ch.label, ch.l) == (rec.radial_label, rec.l)


# ---------------------------------------------------------------------------
# TEST-061-05 (D5-D8): fail-closed manifest diagnostics
# ---------------------------------------------------------------------------


class TestFailClosedD5toD8:
    @pytest.mark.parametrize(
        "mutate, expect",
        [
            pytest.param(
                lambda m: m.update({"schema": "someone_elses_manifest"}),
                ["someone_elses_manifest", "lawaf_trial_basis_manifest"],
                id="D5-unknown-schema",
            ),
            pytest.param(
                lambda m: m.update({"version": "9.9"}),
                ["9.9", "1.0"],
                id="D5-unknown-version",
            ),
            pytest.param(
                lambda m: m["species"][0].pop("sha256"),
                ["sha256"],
                id="D5-missing-fingerprint",
            ),
        ],
    )
    def test_d5_schema_version_fingerprint(self, tmp_path, mutate, expect):
        path = _write_default(tmp_path)
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))
        with pytest.raises(DatasetError) as exc:
            parse_manifest(path)
        for token in expect:
            assert token in str(exc.value)

    def test_d6_fingerprint_mismatch(self, tmp_path):
        path = _write_default(tmp_path)
        data = json.loads(path.read_text())
        data["species"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        with pytest.raises(DatasetError) as exc:
            parse_manifest(path, verify_fingerprints=True)
        msg = str(exc.value)
        assert "0" * 64 in msg and GA_SHA in msg and "Ga" in msg

    def test_d7_harmonic_convention_undeclared(self, tmp_path):
        path = _write_default(tmp_path)
        data = json.loads(path.read_text())
        data.pop("harmonic_convention")
        path.write_text(json.dumps(data))
        with pytest.raises(DatasetError) as exc:
            parse_manifest(path)
        msg = str(exc.value)
        assert "harmonic_convention" in msg

    @pytest.mark.parametrize(
        "trial_definition",
        [None, "pseudo_radial", "hydrogenic"],
    )
    def test_d8_paw_without_projector_kind(self, tmp_path, trial_definition):
        path = _write_default(tmp_path, trial_definition=trial_definition)
        data = json.loads(path.read_text())
        data["species"][0]["kind"] = "paw_xml"
        data["species"][0]["dataset_path"] = str(
            REPO_ROOT / "abinit/tests/Pspdir/Psdj_paw_pbe_std/Ga.xml"
        )
        path.write_text(json.dumps(data))
        with pytest.raises(DatasetError) as exc:
            parse_manifest(path)
        msg = str(exc.value)
        assert "paw" in msg.lower() and "dual_projector" in msg


@pytest.mark.parametrize("normalization", [None, "", 7])
def test_missing_or_ambiguous_harmonic_normalization_is_rejected(
    tmp_path, normalization
):
    path = _write_default(tmp_path)
    data = json.loads(path.read_text())
    if normalization is None:
        data["harmonic_convention"].pop("normalization")
    else:
        data["harmonic_convention"]["normalization"] = normalization
    path.write_text(json.dumps(data))
    with pytest.raises(DatasetError, match="harmonic_convention") as error:
        parse_manifest(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("spin_labels", [None, "false", 1, []])
def test_spin_frame_labels_require_explicit_boolean(tmp_path, spin_labels):
    path = _write_default(tmp_path)
    data = json.loads(path.read_text())
    if spin_labels is None:
        data.pop("spin_frame_labels")
    else:
        data["spin_frame_labels"] = spin_labels
    path.write_text(json.dumps(data))
    with pytest.raises(DatasetError, match="spin_frame_labels") as error:
        parse_manifest(path)
    assert str(path) in str(error.value)


def test_accepted_manifest_cannot_mutate_operator_or_dataset_identity(tmp_path):
    manifest = parse_manifest(_write_default(tmp_path))
    for mapping, key, replacement in (
        (manifest.units, "length", "angstrom"),
        (manifest.harmonic_convention, "normalization", "unnormalized"),
        (manifest.species_fingerprints, "Ga", "0" * 64),
        (manifest.dataset_paths, "Ga", "another.upf"),
    ):
        with pytest.raises(TypeError):
            mapping[key] = replacement
    with pytest.raises(TypeError):
        manifest.harmonic_convention["local_axes"][0][0] = 100


@pytest.mark.parametrize(
    "l,h", [(-1, 0), (8, 0), (True, 0), (0, -1), (0, 1), (1, 3), (1, 0.5)]
)
def test_angular_labels_are_declared_valid_slots(tmp_path, l, h):
    path = _write_default(tmp_path)
    data = json.loads(path.read_text())
    data["species"][0]["radial_ids"][0].update(l=l, harmonics=[h])
    path.write_text(json.dumps(data))
    with pytest.raises(DatasetError) as error:
        parse_manifest(path)
    assert str(path) in str(error.value)
    assert "radial[PP_CHI.3]" in str(error.value)
