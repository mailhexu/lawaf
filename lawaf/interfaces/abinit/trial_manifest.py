# SPDX-License-Identifier: BSD-2-Clause
"""Trial-basis manifest: schema, parser and default NC manifest writer.

Story 061 (lawaf abinit interface), Architecture §4.1/§5/§9.  The manifest is
a small JSON text file (schema ``lawaf_trial_basis_manifest``) declaring
version, species dataset references + SHA256 fingerprints, trial site /
radial / harmonic / spin labels, units and the pseudo-versus-PAW trial
definition kind.  It never duplicates radial arrays already present in the
UPF/XML datasets.  Unknown schema/version or missing fingerprints (D5),
fingerprint mismatches (D6), duplicate/ambiguous trial labels or an
undeclared harmonic convention (D7) and PAW manifests without an explicit
projector definition kind (D8) all fail closed with :class:`DatasetError`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lawaf.interfaces.abinit.basis_datasets import DatasetError, sha256_file

__all__ = [
    "TrialManifest",
    "TrialRecord",
    "TrialSite",
    "parse_manifest",
    "default_nc_manifest",
]

SCHEMA_NAME = "lawaf_trial_basis_manifest"
SCHEMA_VERSION = "1.0"
DEFAULT_HARMONICS = {0: [0], 1: [0, 1, 2], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6]}
L_LETTER = "spdfg"
SPIN_FRAMES = ("up", "down")


def _fail(file, field, expected, detail="") -> DatasetError:
    msg = f"{file}: field {field!r}: expected {expected}. {detail}".rstrip()
    return DatasetError(msg)


@dataclass(frozen=True)
class TrialRecord:
    """One ordered trial declaration; a reference, never a radial array."""

    site_index: int
    species: str
    radial_id: str
    radial_label: str
    l: int
    harmonic_index: int
    harmonic_label: str
    spin_label: Optional[str]


@dataclass(frozen=True)
class TrialManifest:
    path: str
    version: str
    units: Dict[str, str]
    harmonic_convention: Dict[str, Any]
    trial_definition: str
    spin_frame_labels: bool
    species_fingerprints: Dict[str, str]
    trial_records: Tuple[TrialRecord, ...]
    dataset_paths: Dict[str, str]
    sites: Tuple["TrialSite", ...]
    position_coordinates: str

    def __post_init__(self):
        for field in (
            "units",
            "harmonic_convention",
            "species_fingerprints",
            "dataset_paths",
        ):
            object.__setattr__(self, field, _freeze(getattr(self, field)))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class TrialSite:
    label: str
    species: str
    position: Tuple[float, float, float]
    local_axes: Tuple[Tuple[float, float, float], ...]


def _expand_records(data: Dict[str, Any], path: str) -> Tuple[TrialRecord, ...]:
    species = {s["name"]: s for s in data["species"]}
    records: List[TrialRecord] = []
    seen = set()
    for site_index, site in enumerate(data["sites"]):
        sp = species[site["species"]]
        for radial in sp["radial_ids"]:
            l = radial.get("l")
            if type(l) is not int or not 0 <= l < len(L_LETTER):
                raise _fail(
                    path,
                    f"radial[{radial.get('id')}].l",
                    f"integer angular momentum0..{len(L_LETTER)-1}",
                    f"found{l!r}",
                )
            for h in radial["harmonics"]:
                if type(h) is not int or not 0 <= h < 2 * l + 1:
                    raise _fail(
                        path,
                        f"radial[{radial.get('id')}].harmonics",
                        f"integer harmonic slots0..{2*l} for l={l}",
                        f"found{h!r}",
                    )
                spatial_key = (site["label"], radial["id"], int(h))
                if spatial_key in seen:
                    raise _fail(
                        path,
                        "trial label",
                        "unique (site, radial, harmonic) trial labels",
                        f"duplicate trial label from site {site['label']!r}, "
                        f"radial {radial['id']!r}, harmonic {h}",
                    )
                seen.add(spatial_key)
                spins = SPIN_FRAMES if data["spin_frame_labels"] else (None,)
                for spin in spins:
                    records.append(
                        TrialRecord(
                            site_index=site_index,
                            species=sp["name"],
                            radial_id=radial["id"],
                            radial_label=radial.get("label", ""),
                            l=l,
                            harmonic_index=int(h),
                            harmonic_label=L_LETTER[l],
                            spin_label=spin,
                        )
                    )
    return tuple(records)


def parse_manifest(path, verify_fingerprints: bool = True) -> TrialManifest:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise _fail(
            str(path),
            "manifest text",
            f"valid JSON for schema {SCHEMA_NAME!r}",
            str(exc),
        ) from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA_NAME:
        declared = data.get("schema") if isinstance(data, dict) else type(data).__name__
        raise _fail(
            str(path),
            "schema",
            f"{SCHEMA_NAME!r}",
            f"unknown manifest schema {declared!r}",
        )
    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise _fail(
            str(path),
            "version",
            SCHEMA_VERSION,
            f"unknown manifest schema version {version!r}",
        )
    for field in (
        "units",
        "harmonic_convention",
        "trial_definition",
        "species",
        "sites",
    ):
        if field not in data:
            raise _fail(str(path), field, f"the required {field!r} declaration")
    convention = data["harmonic_convention"]
    if (
        not isinstance(convention, dict)
        or not convention.get("name")
        or not isinstance(convention.get("normalization"), str)
        or not convention["normalization"].strip()
    ):
        raise _fail(
            str(path),
            "harmonic_convention",
            "a declared named convention/normalization (e.g. "
            "real_spherical_harmonics with explicit normalization and local axes)",
            "the harmonic convention is undeclared",
        )
    if not isinstance(data.get("spin_frame_labels"), bool):
        raise _fail(path, "spin_frame_labels", "an explicit boolean declaration")

    species = data["species"]
    fingerprints: Dict[str, str] = {}
    dataset_paths: Dict[str, str] = {}
    for sp in species:
        for field in ("name", "kind", "dataset_path", "sha256", "radial_ids"):
            if field not in sp:
                raise _fail(
                    str(path),
                    f"species.{field}",
                    f"the required {field!r} entry",
                    f"species entry {sp.get('name', '?')!r} is incomplete",
                )
        fingerprints[sp["name"]] = sp["sha256"]
        dataset = Path(sp["dataset_path"])
        if not dataset.is_absolute():
            dataset = path.parent / dataset
        dataset_paths[sp["name"]] = str(dataset.resolve())
        if sp["kind"].startswith("paw"):
            if data["trial_definition"] not in (
                "paw_dual_projector",
                "paw_augmented_partial_wave",
            ):
                raise _fail(
                    str(path),
                    "trial_definition",
                    "explicit paw_dual_projector or paw_augmented_partial_wave radial trial definition",
                    f"PAW species {sp['name']!r} declared without an explicit "
                    "radial trial definition kind "
                    f"(trial_definition={data['trial_definition']!r})",
                )
        if verify_fingerprints:
            actual = sha256_file(dataset_paths[sp["name"]])
            if actual != sp["sha256"]:
                raise _fail(
                    str(path),
                    f"species[{sp['name']!r}].sha256",
                    sp["sha256"],
                    f"dataset {sp['dataset_path']} fingerprint mismatch "
                    f"(actual {actual})",
                )

    coordinates = data.get("position_coordinates")
    if coordinates not in ("reduced", "cartesian"):
        raise _fail(
            path, "position_coordinates", "explicit reduced or cartesian coordinates"
        )
    sites = []
    labels = set()
    for index, site in enumerate(data["sites"]):
        label = site.get("label")
        if not label or label in labels or site.get("species") not in fingerprints:
            raise _fail(
                path,
                f"sites[{index}]",
                "unique label and a declared species",
                f"label={label!r}, species={site.get('species')!r}",
            )
        labels.add(label)
        position = np.asarray(site.get("position"), dtype=float)
        axes = np.asarray(site.get("local_axes"), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise _fail(path, f"sites[{index}].position", "three finite coordinates")
        if (
            axes.shape != (3, 3)
            or not np.all(np.isfinite(axes))
            or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-12, rtol=0)
            or np.linalg.det(axes) < 0
        ):
            raise _fail(
                path,
                f"sites[{index}].local_axes",
                "right-handed orthonormal Cartesian axes in rows",
            )
        sites.append(
            TrialSite(label, site["species"], tuple(position), tuple(map(tuple, axes)))
        )
    return TrialManifest(
        path=str(path),
        version=version,
        units=dict(data["units"]),
        harmonic_convention=dict(convention),
        trial_definition=data["trial_definition"],
        spin_frame_labels=data["spin_frame_labels"],
        species_fingerprints=fingerprints,
        trial_records=_expand_records(data, str(path)),
        dataset_paths=dataset_paths,
        sites=tuple(sites),
        position_coordinates=coordinates,
    )


def default_nc_manifest(
    path,
    species_datasets: Sequence[Tuple[str, Path]],
    sites: Sequence[Dict[str, Any]],
    spin_frame_labels: bool = False,
    trial_definition: Optional[str] = "pseudo_radial",
) -> Path:
    """Write the Architecture §4.1/§9 default NC manifest.

    Selects radial IDs ``PP_CHI.3(4S,l0)`` and ``PP_CHI.4(4P,l1)`` per
    species, one s harmonic plus three declared real p harmonics per atom
    (8 scalar trials for the 2-atom GaAs cell; 16 with both spin-frame
    labels).  The manifest carries labels and fingerprints only.
    """
    species = []
    for name, dataset in species_datasets:
        species.append(
            {
                "name": name,
                "kind": "nc_upf",
                "dataset_path": str(Path(dataset).resolve()),
                "sha256": sha256_file(dataset),
                "radial_ids": [
                    {
                        "id": "PP_CHI.3",
                        "label": "4S",
                        "l": 0,
                        "harmonics": list(DEFAULT_HARMONICS[0]),
                    },
                    {
                        "id": "PP_CHI.4",
                        "label": "4P",
                        "l": 1,
                        "harmonics": list(DEFAULT_HARMONICS[1]),
                    },
                ],
            }
        )
    normalized_sites = []
    for i, site in enumerate(sites):
        normalized_sites.append(
            {
                "label": site.get("label", f"site{i}"),
                "species": site["species"],
                "position": list(site["position"]),
                "local_axes": site.get("local_axes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            }
        )
    doc = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": "real_spherical_harmonics_wannier90",
            "normalization": "normalized",
            "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "trial_definition": trial_definition,
        "spin_frame_labels": spin_frame_labels,
        "species": species,
        "sites": normalized_sites,
    }
    path = Path(path)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
