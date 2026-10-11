# SPDX-License-Identifier: BSD-2-Clause
"""Projector/radial dataset readers for NC UPF and ABINIT PAW-XML datasets.

Story 061 (lawaf abinit interface).  The readers return immutable dataset
records and keep radial arrays exactly as stored on the dataset grid: no
resampling, no double r factors, no R/rR relabeling.  j-coupled labels come
only from parsed ``PP_SPIN_ORB``/``PP_RELWFC`` metadata; occupations are used
solely as an asserted cross-check, never to infer quantum numbers.  Every
structural inconsistency fails closed with a :class:`DatasetError` naming the
file, the field and the expected value.

Independent implementation from the documented UPF 2.0.1 and ABINIT
PAW-XML v0.7 format contracts (stdlib ``xml.etree.ElementTree`` + numpy
only; no pypao/abinao import).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DatasetError",
    "RadialGrid",
    "UpfChannel",
    "UpfGrid",
    "UpfProjectorDataset",
    "SpinOrbitPartner",
    "PawChannel",
    "PawGrid",
    "PawXmlDataset",
    "sha256_file",
    "read_upf_projectors",
    "read_paw_xml_dataset",
]

SUPPORTED_UPF_VERSIONS = ("2.0.1",)
SUPPORTED_PAW_XML_VERSIONS = ("0.7",)


class DatasetError(Exception):
    """Fail-closed dataset/manifest error naming file, field and expectation."""


def _fail(file, field, expected, detail="") -> DatasetError:
    msg = f"{file}: field {field!r}: expected {expected}. {detail}".rstrip()
    return DatasetError(msg)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _floats(text: Optional[str]) -> np.ndarray:
    values = np.array(text.split() if text is not None else [], dtype=np.float64)
    values.setflags(write=False)
    return values


# ---------------------------------------------------------------------------
# NC UPF reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpfGrid:
    """The UPF mesh exactly as stored: ``r`` (PP_R) and quadrature (PP_RAB)."""

    representation: str  # "r": arrays are tabulated as plain r factors
    r: np.ndarray
    quadrature_weights: np.ndarray  # PP_RAB, d r / d i as stored

    def __len__(self) -> int:
        return len(self.r)


@dataclass(frozen=True)
class UpfChannel:
    """One ``PP_CHI.n`` radial channel in file order, values as stored."""

    radial_id: str  # e.g. "PP_CHI.3"
    label: str  # e.g. "4S"
    l: int
    values: np.ndarray
    grid: UpfGrid
    occupation: float


@dataclass(frozen=True)
class SpinOrbitPartner:
    """One ``PP_RELWFC.n`` j-partner record parsed from ``PP_SPIN_ORB``."""

    index: int  # 1-based, matches the PP_CHI file index
    lchi: int
    jchi: float
    nn: int


@dataclass(frozen=True)
class UpfProjectorDataset:
    path: str
    element: str
    z_valence: float
    version: str
    grid: UpfGrid
    channels: Tuple[UpfChannel, ...]
    _spin_orbit: Optional[Tuple[SpinOrbitPartner, ...]]

    def get_channel(self, radial_id: str) -> UpfChannel:
        for ch in self.channels:
            if ch.radial_id == radial_id:
                return ch
        available = ", ".join(ch.radial_id for ch in self.channels)
        raise _fail(
            self.path,
            "radial_id",
            f"one of [{available}]",
            f"requested radial ID {radial_id!r} is absent from the dataset",
        )

    def spin_orbit_partners(self) -> Tuple[SpinOrbitPartner, ...]:
        if self._spin_orbit is None:
            raise _fail(
                self.path,
                "PP_SPIN_ORB/PP_RELWFC",
                "the spin-orbit metadata block present in the UPF",
                "missing PP_SPIN_ORB/PP_RELWFC block: j-coupled trial labels "
                "require the file's explicit jchi/nn metadata and cannot be "
                "guessed; re-generate or fetch a fully relativistic UPF with "
                "spin-orbit data",
            )
        return self._spin_orbit

    def assert_occupation_consistency(self) -> None:
        """Assert-only cross-check of parsed jchi against occupations.

        Within each (label, l) group of channels, the occupation per m state
        (occupation / (2j+1)) must be equal across the j partners.  This is
        never consulted when resolving j labels.
        """
        partners = {p.index: p for p in self.spin_orbit_partners()}
        groups: dict = {}
        for i, ch in enumerate(self.channels, start=1):
            if ch.l != partners[i].lchi:
                raise _fail(
                    self.path,
                    f"PP_RELWFC.{i}.lchi",
                    str(ch.l),
                    f"cross-check: lchi {partners[i].lchi} disagrees with the "
                    f"PP_CHI.{i} angular momentum",
                )
            groups.setdefault((ch.label, ch.l), []).append((i, ch, partners[i]))
        for (label, l), members in groups.items():
            perm = [ch.occupation / (2.0 * p.jchi + 1.0) for _, ch, p in members]
            ref = perm[0]
            for (i, _, p), value in zip(members, perm):
                if value <= 0.0 or abs(value - ref) > 0.01 * ref:
                    raise _fail(
                        self.path,
                        f"PP_CHI.{i}.occupation",
                        f"{ref:.4g} electrons per m (from {label}, l={l}, "
                        f"j={p.jchi})",
                        f"cross-check failed: per-m occupation {value:.4g}",
                    )


def _parse_text_array(el, size: int, path: str, field: str) -> np.ndarray:
    values = _floats(el.text)
    if len(values) != size:
        raise _fail(
            path,
            field,
            f"{size} values (declared size attribute)",
            f"found {len(values)} values",
        )
    return values


def read_upf_projectors(path, sha256: Optional[str] = None) -> UpfProjectorDataset:
    path = Path(path)
    if sha256 is not None:
        actual = sha256_file(path)
        if actual != sha256:
            raise _fail(
                str(path),
                "sha256",
                sha256,
                f"dataset fingerprint mismatch (actual {actual})",
            )
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise _fail(str(path), "XML", "a well-formed UPF document", str(exc)) from exc
    if root.tag != "UPF":
        raise _fail(str(path), "root element", "UPF", f"found {root.tag!r}")
    version = root.get("version", "")
    if version not in SUPPORTED_UPF_VERSIONS:
        raise _fail(
            str(path),
            "UPF version",
            "one of " + ", ".join(SUPPORTED_UPF_VERSIONS),
            f"declared version {version!r} is not supported",
        )
    header = root.find("PP_HEADER")
    if header is None:
        raise _fail(str(path), "PP_HEADER", "the UPF header block")
    element = header.get("element", "").strip()
    if header.get("pseudo_type", "NC").strip() != "NC":
        raise _fail(
            str(path),
            "PP_HEADER@pseudo_type",
            "NC",
            f"found {header.get('pseudo_type')!r}; this reader covers NC UPF "
            "(PAW datasets use the PAW-XML reader)",
        )

    mesh = root.find("./PP_MESH")
    if mesh is None:
        raise _fail(str(path), "PP_MESH", "the radial mesh block")
    pp_r = mesh.find("PP_R")
    pp_rab = mesh.find("PP_RAB")
    if pp_r is None or pp_rab is None:
        raise _fail(str(path), "PP_MESH/PP_R|PP_RAB", "mesh and quadrature arrays")
    size = int(pp_r.get("size", "0"))
    r = _parse_text_array(pp_r, size, str(path), "PP_R")
    rab = _parse_text_array(pp_rab, size, str(path), "PP_RAB")
    if not np.all(np.diff(r) > 0):
        raise _fail(str(path), "PP_R", "a strictly monotonically increasing grid")
    grid = UpfGrid(representation="r", r=r, quadrature_weights=rab)

    channels = []
    for index in range(1, 1000):
        el = root.find(f"./PP_PSWFC/PP_CHI.{index}")
        if el is None:
            el = next(root.iter(f"PP_CHI.{index}"), None)
        if el is None:
            break
        ch_size = int(el.get("size", "0"))
        if ch_size != size:
            raise _fail(
                str(path),
                f"PP_CHI.{index}@size",
                str(size),
                f"channel grid length {ch_size} inconsistent with the mesh",
            )
        values = _parse_text_array(el, ch_size, str(path), f"PP_CHI.{index} values")
        channels.append(
            UpfChannel(
                radial_id=f"PP_CHI.{index}",
                label=el.get("label", "").strip(),
                l=int(el.get("l", "-1")),
                values=values,
                grid=grid,
                occupation=float(el.get("occupation", "nan")),
            )
        )
    if not channels:
        raise _fail(str(path), "PP_CHI", "at least one pseudo-wavefunction channel")

    so_block = root.find("PP_SPIN_ORB")
    partners: Optional[Tuple[SpinOrbitPartner, ...]] = None
    if so_block is not None:
        parsed = []
        for index in range(1, len(channels) + 1):
            el = so_block.find(f"PP_RELWFC.{index}")
            if el is None:
                break
            parsed.append(
                SpinOrbitPartner(
                    index=index,
                    lchi=int(el.get("lchi", "-1")),
                    jchi=float(el.get("jchi", "nan")),
                    nn=int(el.get("nn", "0")),
                )
            )
        if len(parsed) == len(channels):
            partners = tuple(parsed)

    return UpfProjectorDataset(
        path=str(path),
        element=element,
        z_valence=float(header.get("z_valence", "nan")),
        version=version,
        grid=grid,
        channels=tuple(channels),
        _spin_orbit=partners,
    )


# ---------------------------------------------------------------------------
# PAW XML reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PawGrid:
    r: np.ndarray
    equation: str  # e.g. "r=a*(exp(d*i)-1)"
    a: float
    d: float
    id: str


@dataclass(frozen=True)
class PawChannel:
    """One (n, l) PAW channel: AE/pseudo partial wave and dual projector."""

    state_id: str
    n: int
    l: int
    rc: float
    occupation: float
    ae_partial_wave: np.ndarray  # R(r), without the radial Jacobian, as stored
    pseudo_partial_wave: np.ndarray
    dual_projector: np.ndarray


@dataclass(frozen=True)
class PawXmlDataset:
    path: str
    element: str
    version: str
    radial_representation: str  # "R": radial contractions use r^2 dr
    rc: float  # paw_radius@rc
    grid: PawGrid
    channels: Tuple[PawChannel, ...]


def read_paw_xml_dataset(path, sha256: Optional[str] = None) -> PawXmlDataset:
    path = Path(path)
    if sha256 is not None:
        actual = sha256_file(path)
        if actual != sha256:
            raise _fail(
                str(path),
                "sha256",
                sha256,
                f"dataset fingerprint mismatch (actual {actual})",
            )
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise _fail(
            str(path), "XML", "a well-formed PAW-XML document", str(exc)
        ) from exc
    if root.tag != "paw_dataset":
        raise _fail(str(path), "root element", "paw_dataset", f"found {root.tag!r}")
    version = root.get("version", "")
    if version not in SUPPORTED_PAW_XML_VERSIONS:
        raise _fail(
            str(path),
            "paw_dataset version",
            "one of " + ", ".join(SUPPORTED_PAW_XML_VERSIONS),
            f"declared version {version!r} is not supported",
        )

    rg = root.find("radial_grid")
    if rg is None:
        raise _fail(str(path), "radial_grid", "the radial grid block")
    values_el = rg.find("values")
    if values_el is None:
        raise _fail(str(path), "radial_grid/values", "the grid values")
    r = _floats(values_el.text)
    if len(r) == 0:
        raise _fail(str(path), "radial_grid/values", "a non-empty grid")
    if not np.all(np.diff(r) > 0):
        raise _fail(
            str(path), "radial_grid/values", "a strictly monotonically increasing grid"
        )
    grid = PawGrid(
        r=r,
        equation=rg.get("eq", "").strip(),
        a=float(rg.get("a", "nan")),
        d=float(rg.get("d", "nan")),
        id=rg.get("id", "").strip(),
    )
    n = len(r)

    paw_radius = root.find("paw_radius")
    if paw_radius is None or paw_radius.get("rc") is None:
        raise _fail(str(path), "paw_radius@rc", "the augmentation radius metadata")
    rc = float(paw_radius.get("rc"))

    states = root.findall("./valence_states/state")
    if not states:
        raise _fail(str(path), "valence_states", "at least one state entry")
    state_meta = {}
    for st in states:
        sid = st.get("id", "").strip()
        state_meta[sid] = {
            "n": int(st.get("n", "0")),
            "l": int(st.get("l", "-1")),
            "rc": float(st.get("rc", "nan")),
            "f": float(st.get("f", "nan")),
        }

    def _array(tag: str, sid: str) -> np.ndarray:
        for el in root.findall(tag):
            if el.get("state", "").strip() == sid:
                values = _floats(el.text)
                if len(values) != n:
                    raise _fail(
                        str(path),
                        f"{tag}[{sid}]",
                        f"{n} values on the dataset grid",
                        f"found {len(values)} values",
                    )
                return values
        raise _fail(str(path), f"{tag}[{sid}]", "the channel array")

    channels = []
    for sid, meta in state_meta.items():
        channels.append(
            PawChannel(
                state_id=sid,
                n=meta["n"],
                l=meta["l"],
                rc=meta["rc"],
                occupation=meta["f"],
                ae_partial_wave=_array("ae_partial_wave", sid),
                pseudo_partial_wave=_array("pseudo_partial_wave", sid),
                dual_projector=_array("projector_function", sid),
            )
        )
    return PawXmlDataset(
        path=str(path),
        element=root.find("./atom").get("symbol", "").strip()
        if root.find("./atom") is not None
        else "",
        version=version,
        radial_representation="R",
        rc=rc,
        grid=grid,
        channels=tuple(channels),
    )
