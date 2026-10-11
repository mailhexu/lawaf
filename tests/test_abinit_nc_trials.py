"""Numerical trials checked against independent Cartesian defining integrals."""

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from lawaf.interfaces.abinit.basis_datasets import DatasetError, sha256_file
from lawaf.interfaces.abinit.nc_trials import NcTrialBasis
from lawaf.interfaces.abinit.trial_manifest import parse_manifest


def _gaussian_basis(tmp_path):
    radial = np.linspace(0, 12, 2401)
    root = ET.Element("UPF", version="2.0.1")
    ET.SubElement(root, "PP_HEADER", element="H", pseudo_type="NC", z_valence="1")
    mesh = ET.SubElement(root, "PP_MESH")
    for name, values in (("PP_R", radial), ("PP_RAB", np.full_like(radial, 0.005))):
        ET.SubElement(mesh, name, size=str(len(radial))).text = " ".join(
            map(str, values)
        )
    waves = ET.SubElement(root, "PP_PSWFC")
    for l in range(4):
        ET.SubElement(
            waves,
            f"PP_CHI.{l+1}",
            size=str(len(radial)),
            l=str(l),
            label=f"{l+1}spdf"[0:1] + "spdf"[l],
            occupation="0",
        ).text = " ".join(map(str, radial ** (l + 1) * np.exp(-(radial**2) / 2)))
    upf = tmp_path / "gaussian.upf"
    ET.ElementTree(root).write(upf)
    axes = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    document = {
        "schema": "lawaf_trial_basis_manifest",
        "version": "1.0",
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": "real_spherical_harmonics_wannier90",
            "normalization": "normalized",
        },
        "trial_definition": "pseudo_radial",
        "spin_frame_labels": False,
        "species": [
            {
                "name": "H",
                "kind": "nc_upf",
                "dataset_path": upf.name,
                "sha256": sha256_file(upf),
                "radial_ids": [
                    {"id": f"PP_CHI.{l+1}", "l": l, "harmonics": list(range(2 * l + 1))}
                    for l in range(4)
                ],
            }
        ],
        "sites": [
            {
                "label": "H0",
                "species": "H",
                "position": [0.17, -0.09, 0.23],
                "local_axes": axes,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    header = SimpleNamespace(
        rprimd=np.array([[10.0, 0, 0], [2, 11, 0], [0, 1, 9]]),
        kpt=np.array([[0.13, -0.22, 0.09], [0, 0, 0]]),
        usepaw=0,
    )
    return path, document, header


def _solid_harmonics(points):
    x, y, z = points.T
    r2 = x * x + y * y + z * z
    p = np.pi
    return np.column_stack(
        (
            np.ones_like(x) / np.sqrt(4 * p),
            np.sqrt(3 / (4 * p)) * z,
            np.sqrt(3 / (4 * p)) * x,
            np.sqrt(3 / (4 * p)) * y,
            np.sqrt(5 / (16 * p)) * (3 * z * z - r2),
            np.sqrt(15 / (4 * p)) * x * z,
            np.sqrt(15 / (4 * p)) * y * z,
            np.sqrt(15 / (16 * p)) * (x * x - y * y),
            np.sqrt(15 / (4 * p)) * x * y,
            np.sqrt(7 / (16 * p)) * z * (5 * z * z - 3 * r2),
            np.sqrt(21 / (32 * p)) * x * (5 * z * z - r2),
            np.sqrt(21 / (32 * p)) * y * (5 * z * z - r2),
            np.sqrt(105 / (4 * p)) * x * y * z,
            np.sqrt(105 / (16 * p)) * z * (x * x - y * y),
            np.sqrt(35 / (32 * p)) * x * (x * x - 3 * y * y),
            np.sqrt(35 / (32 * p)) * y * (3 * x * x - y * y),
        )
    )


@pytest.mark.parametrize("coordinates", ["reduced", "cartesian"])
def test_native_radial_transform_matches_cartesian_s_p_d_f_integrals(
    tmp_path, coordinates
):
    path, document, header = _gaussian_basis(tmp_path)
    if coordinates == "cartesian":
        document["position_coordinates"] = coordinates
        document["sites"][0]["position"] = (
            np.asarray(document["sites"][0]["position"]) @ header.rprimd
        ).tolist()
        path.write_text(json.dumps(document))
    basis = NcTrialBasis(parse_manifest(path), header)
    kg = np.array([[0, 0, 0], [1, -2, 0], [-1, 0, 2], [2, 1, -1]])
    nodes, weights = hermgauss(24)
    xyz = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(
        -1, 3
    ) * np.sqrt(2)
    measure = (
        weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    ).reshape(-1) * 2**1.5
    site = document["sites"][0]
    localized = _solid_harmonics(xyz @ np.asarray(site["local_axes"]).T)
    centre = np.asarray(site["position"])
    if coordinates == "reduced":
        centre = centre @ header.rprimd
    reciprocal = 2 * np.pi * np.linalg.inv(header.rprimd).T
    for ik in range(2):
        momentum = (kg + header.kpt[ik]) @ reciprocal
        phases = np.exp(-1j * ((xyz + centre) @ momentum.T))
        expected = (
            phases.T
            @ (measure[:, None] * localized)
            / np.sqrt(abs(np.linalg.det(header.rprimd)))
        )
        np.testing.assert_allclose(basis(ik, kg), expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_coordinates", "unknown"),
        ("local_axes", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        ("duplicate_site", None),
    ],
)
def test_manifest_rejects_ambiguous_coordinates_and_left_handed_axes(
    tmp_path, field, value
):
    path, document, _ = _gaussian_basis(tmp_path)
    if field == "local_axes":
        document["sites"][0][field] = value
    elif field == "duplicate_site":
        document["sites"].append(dict(document["sites"][0]))
    else:
        document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(DatasetError):
        parse_manifest(path)
