# SPDX-License-Identifier: BSD-2-Clause
"""Bounded ABINIT WFK reader and eigenvalue accessor (story 060).

Implements the Architecture §3 source seam for ABINIT WFK files in both
storage flavours:

* Fortran unformatted sequential records (``iomode`` 0/1 MPI record format,
  the layout written by ``m_wfk.F90``/``m_hdr.F90``, headform 80);
* native ABINIT NetCDF (``iomode`` 3 / ETSF-named variables), read through
  netCDF4 slab access.

Decode rules are the epic-048-c certified conventions: ``set_istwfk`` bit
semantics (``m_cgtools.F90:898-928``), the packed half-sphere involution
``u_{G0/2}(G) = conj(u_{G0/2}(-G-G0))`` and the storage weights certified
against ``sqnorm_g``/``dotprod_g`` (weight 2 per stored plane wave, weight 1
for the Gamma G=0 coefficient and for self-conjugate planes 2G == -G0).

Energies are returned exactly as stored, in Hartree, with no unit
conversion; the single conversion point is the producer (Architecture §4.1).

No Wannier90 library, binding, ctypes or f90wrap is used anywhere.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

__all__ = [
    "WfkReaderError",
    "WfkHeader",
    "WfkReader",
    "resolve_istwfk",
    "cross_format_check",
    "decode_packed_sphere",
    "SUPPORTED_ABINIT_MAJOR_VERSIONS",
]

SUPPORTED_ABINIT_MAJOR_VERSIONS = (9, 10)
_TOL10 = 1e-10
_NATIVE_PACKED_RESTRICTION = "m_inkpts.F90:501-506"


class WfkReaderError(Exception):
    """Fail-closed WFK reader diagnostic naming file, field and expectation."""

    def __init__(self, diagnostic, path, field, detail):
        self.diagnostic = diagnostic
        self.path = str(path)
        self.field = field
        self.detail = detail
        super().__init__(f"[{diagnostic}] file={path} field={field}: {detail}")


def resolve_istwfk(raw, kpt, *, nspinor=1, pawspnorb=False):
    """Resolve an istwfk value for one k-point (FR-019).

    ``raw == 0`` (automatic) resolves per point with the certified
    ``set_istwfk`` bit semantics.  Anything outside {0..9} or a resolved
    value outside {1..9} is a D3 diagnostic.
    """
    if not (0 <= int(raw) <= 9):
        raise WfkReaderError(
            "D3",
            "<k-point>",
            "istwfk",
            f"unresolved istwfk raw value {int(raw)} (raw), legal values are 0-9 "
            "with 0 resolved per point to 1-9",
        )
    raw = int(raw)
    if raw == 0 and (nspinor != 1 or pawspnorb):
        return 1  # native inkpts forcing precedes automatic special-point packing
    if raw != 0:
        return raw
    k = np.asarray(kpt, dtype=float)
    bit0 = 1
    bits = [0, 0, 0]
    for i in range(3):
        if abs(k[i]) < _TOL10:
            bits[i] = 0
        elif abs(k[i] - 0.5) < _TOL10:
            bits[i] = 1
        else:
            bit0 = 0
    if bit0 == 0:
        return 1
    resolved = 2 + bits[0] + 4 * bits[1] + 2 * bits[2]
    if not (1 <= resolved <= 9):  # pragma: no cover - structural guard
        raise WfkReaderError(
            "D3",
            "<k-point>",
            "istwfk",
            f"automatic istwfk raw value {raw} resolved to {resolved} at "
            f"kpt={list(kpt)}; resolved value must be in 1-9",
        )
    return resolved


@dataclass(frozen=True)
class WfkHeader:
    """Immutable WFK header metadata (source units; bohr, Hartree)."""

    codvsn: str
    headform: int
    fform: int
    bantot: int
    date: int
    intxc: int
    ixc: int
    natom: int
    ngfft: np.ndarray
    nkpt: int
    nspden: int
    nspinor: int
    nsppol: int
    nsym: int
    npsp: int
    ntypat: int
    occopt: int
    pertcase: int
    usepaw: int
    ecut: float
    ecutdg: float
    ecutsm: float
    ecut_eff: float
    qptn: np.ndarray
    rprimd: np.ndarray  # rows are the lattice vectors, source bohr
    stmbias: float
    tphysel: float
    tsmear: float
    usewvl: int
    nshiftk_orig: int
    nshiftk: int
    mband: int
    # per-k / per-record arrays
    istwfk: np.ndarray  # resolved values as stored (1-9)
    istwfk_raw: np.ndarray  # raw values (0 marks automatic)
    nband: np.ndarray  # (nkpt*nsppol,)
    npwarr: np.ndarray  # (nkpt,)
    so_psp: np.ndarray  # (npsp,)
    symafm: np.ndarray
    symrel: np.ndarray  # (nsym, 3, 3)
    typat: np.ndarray
    kpt: np.ndarray  # (nkpt, 3) reduced
    occ: np.ndarray
    tnons: np.ndarray
    znucltypat: np.ndarray
    wtk: np.ndarray
    residm: float
    xred: np.ndarray
    etot: float
    fermie: float
    amu: np.ndarray
    kptopt: int
    pawcpxocc: int
    nelect: float
    cellcharge: float
    icoulomb: int
    kptrlatt: np.ndarray
    kptrlatt_orig: np.ndarray
    shiftk_orig: np.ndarray
    shiftk: np.ndarray
    title: tuple
    znuclpsp: np.ndarray
    zionpsp: np.ndarray
    pspso: np.ndarray
    pspdat: np.ndarray
    pspcod: np.ndarray
    pspxc: np.ndarray
    lmn_size: np.ndarray
    md5_pseudos: tuple
    gmet: np.ndarray = field(default=None, repr=False)

    @property
    def resolved_istwfk(self) -> np.ndarray:
        """Resolved per-k istwfk values (1-9)."""
        return self.istwfk

    @property
    def pawspnorb(self) -> bool:
        """True when spin-orbit PAW projectors are present."""
        return bool(np.any(np.asarray(self.pspso) > 1))


def _gmet_from_rprimd(rprimd):
    rinv = np.linalg.inv(np.asarray(rprimd, dtype=float))
    return (2.0 * np.pi) ** 2 * (rinv.T @ rinv)


class WfkReader:
    """Bounded reader for one WFK file.

    The reader owns a single open file handle, exposes immutable header
    metadata and bounded per-``(k, spin, band-window)`` block access.  No
    all-k or full-BZ coefficient collection happens here (Architecture §8
    lawaf ledger, FR-023).
    """

    def __init__(self, path, fmt, handle, header, index):
        self._path = Path(path)
        self._fmt = fmt  # "fortran-record" | "netcdf"
        self._fh = handle
        self._header = header
        self._index = index  # (k, spin) -> record/offset map (binary) or None
        self._live_bytes = 0
        self._peak_live_bytes = 0

    # -- construction -------------------------------------------------------

    @classmethod
    def open(cls, path):
        path = Path(path)
        with open(path, "rb") as probe:
            magic = probe.read(4)
        if magic == b"CDF\x01" or magic[:4] == b"\x89HDF":
            return cls._open_netcdf(path)
        return cls._open_fortran(path)

    # -- Fortran-record flavour ---------------------------------------------

    @classmethod
    def _open_fortran(cls, path):
        fh = open(path, "rb")
        try:
            def record_fail(rec_id, expected, actual):
                raise WfkReaderError(
                    "D5",
                    path,
                    f"record {rec_id}",
                    f"truncated or unreadable record: expected {expected} bytes, "
                    f"found {actual}",
                )
    
            def read_record(pos):
                head = fh.read(4)
                if len(head) != 4:
                    record_fail("header", 4, len(head))
                (n,) = struct.unpack("<I", head)
                payload = fh.read(n)
                if len(payload) != n:
                    record_fail("header", n, len(payload))
                tail = fh.read(4)
                if len(tail) != 4:
                    record_fail("header", 4, len(tail))
                (n2,) = struct.unpack("<I", tail)
                if n2 != n:
                    record_fail("header", n, n2)
                return payload, fh.tell()
    
            codvsn_raw, headform, fform = struct.unpack("<8sii", read_record(0)[0])
            codvsn = codvsn_raw.decode("ascii").strip("\x00 ")
            try:
                major = int(codvsn.split(".")[0])
            except ValueError:
                major = None
            if major not in SUPPORTED_ABINIT_MAJOR_VERSIONS:
                raise WfkReaderError(
                    "D1",
                    path,
                    "codvsn",
                    f"unsupported codvsn {codvsn!r}; supported major versions are "
                    f"{list(SUPPORTED_ABINIT_MAJOR_VERSIONS)}",
                )
            if headform < 40:
                raise WfkReaderError(
                    "D1",
                    path,
                    "headform",
                    f"unsupported header format {headform}; supported headforms >= 40",
                )
    
            p2, _ = read_record(0)
            ints = struct.unpack_from("<18i", p2, 0)
            (bantot, date, intxc, ixc, natom) = ints[:5]
            ngfft = np.array(ints[5:8])
            (
                nkpt,
                nspden,
                nspinor,
                nsppol,
                nsym,
                npsp,
                ntypat,
                occopt,
                pertcase,
                usepaw,
            ) = ints[8:18]
            dbl = struct.unpack_from("<4d", p2, 72)
            ecut, ecutdg, ecutsm, ecut_eff = dbl
            off = 72 + 32
            qptn = np.array(struct.unpack_from("<3d", p2, off))
            off += 24
            # File order lists ABINIT columns: each consecutive triple is a vector.
            rprimd = np.array(struct.unpack_from("<9d", p2, off)).reshape(3, 3, order="C")
            off += 72
            stmbias, tphysel, tsmear = struct.unpack_from("<3d", p2, off)
            off += 24
            usewvl, nshiftk_orig, nshiftk, mband = struct.unpack_from("<4i", p2, off)
            if len(p2) != off + 16:
                record_fail("header-2", off + 16, len(p2))
    
            p3, _ = read_record(0)
            exp3 = (nkpt + nkpt * nsppol + nkpt + npsp + nsym + 9 * nsym + natom) * 4 + (
                3 * nkpt + mband * nkpt * nsppol + 3 * nsym + natom + nkpt
            ) * 8
            if len(p3) != exp3:
                record_fail("header-karrays", exp3, len(p3))
            pos = 0
    
            def take(dtype, count):
                nonlocal pos
                dt = np.dtype(dtype)
                arr = np.frombuffer(p3, dtype=dt, count=count, offset=pos)
                pos += dt.itemsize * count
                return arr.copy()
    
            istwfk = take("<i4", nkpt)
            nband = take("<i4", nkpt * nsppol)
            npwarr = take("<i4", nkpt)
            so_psp = take("<i4", npsp)
            symafm = take("<i4", nsym)
            symrel = take("<i4", 9 * nsym).reshape(nsym, 3, 3, order="C")
            typat = take("<i4", natom)
            kpt = take("<f8", 3 * nkpt).reshape(nkpt, 3, order="C")
            occ = take("<f8", mband * nkpt * nsppol).reshape(
                mband, nkpt * nsppol, order="F"
            )
            tnons = take("<f8", 3 * nsym).reshape(nsym, 3, order="C")
            znucltypat = take("<f8", natom).astype(np.int32)
            wtk = take("<f8", nkpt)
    
            p4, _ = read_record(0)
            exp4 = (3 * natom + 2 + ntypat) * 8 + 8
            if len(p4) != exp4:
                record_fail("header-atomarrays", exp4, len(p4))
            pos = 0
            (residm,) = struct.unpack_from("<d", p4, 0)
            xred = (
                np.frombuffer(p4, "<f8", 3 * natom, 8).reshape(natom, 3, order="C").copy()
            )
            q = 8 + 3 * natom * 8
            etot, fermie = struct.unpack_from("<2d", p4, q)
            amu = np.frombuffer(p4, "<f8", ntypat, q + 16).copy()
    
            p5, _ = read_record(0)
            (kptopt, pawcpxocc) = struct.unpack_from("<2i", p5, 0)
            nelect, cellcharge = struct.unpack_from("<2d", p5, 8)
            (icoulomb,) = struct.unpack_from("<i", p5, 24)
            q = 28
            kptrlatt = np.frombuffer(p5, "<i4", 9, q).reshape(3, 3, order="F").copy()
            q += 36
            kptrlatt_orig = np.frombuffer(p5, "<i4", 9, q).reshape(3, 3, order="F").copy()
            q += 36
            shiftk_orig = (
                np.frombuffer(p5, "<f8", 3 * nshiftk_orig, q)
                .reshape(nshiftk_orig, 3, order="C")
                .copy()
            )
            q += 24 * nshiftk_orig
            shiftk = (
                np.frombuffer(p5, "<f8", 3 * nshiftk, q)
                .reshape(nshiftk, 3, order="C")
                .copy()
            )
    
            if occopt == 9:
                read_record(0)  # ivalence, ne_qFD, nh_qFD, fermie, fermih
    
            title, znuclpsp, zionpsp = [], [], []
            pspso = np.zeros(npsp, np.int32)
            pspdat = np.zeros(npsp, np.int32)
            pspcod = np.zeros(npsp, np.int32)
            pspxc = np.zeros(npsp, np.int32)
            lmn_size = np.zeros(npsp, np.int32)
            md5_pseudos = []
            for ipsp in range(npsp):
                pp, _ = read_record(0)
                if len(pp) != 132 + 16 + 20 + 32:
                    record_fail(f"header-psp{ipsp + 1}", 200, len(pp))
                title.append(pp[:132].decode("ascii", "replace").strip("\x00 "))
                (
                    z,
                    zion,
                    pspso[ipsp],
                    pspdat[ipsp],
                    pspcod[ipsp],
                    pspxc[ipsp],
                    lmn_size[ipsp],
                ) = struct.unpack_from("<2d5i", pp, 132)
                znuclpsp.append(z)
                zionpsp.append(zion)
                md5_pseudos.append(pp[132 + 36 :].decode("ascii").strip("\x00 "))
            if usepaw == 1:
                # pawrhoij record block: skip; not needed by this reader (story 062
                # owns supplied-coefficient seams).  Fail closed is NOT correct
                # here: PAW WFKs remain readable, rhoij is simply not parsed.
                raise WfkReaderError(
                    "D1",
                    path,
                    "pawrhoij",
                    "PAW (usepaw=1) Fortran-record WFK: pawrhoij record block "
                    "parsing is owned by story 062; this build reads NC scalar and "
                    "SOC WFK files",
                )
    
            header = WfkHeader(
                codvsn=codvsn,
                headform=headform,
                fform=fform,
                bantot=bantot,
                date=date,
                intxc=intxc,
                ixc=ixc,
                natom=natom,
                ngfft=ngfft,
                nkpt=nkpt,
                nspden=nspden,
                nspinor=nspinor,
                nsppol=nsppol,
                nsym=nsym,
                npsp=npsp,
                ntypat=ntypat,
                occopt=occopt,
                pertcase=pertcase,
                usepaw=usepaw,
                ecut=ecut,
                ecutdg=ecutdg,
                ecutsm=ecutsm,
                ecut_eff=ecut_eff,
                qptn=qptn,
                rprimd=rprimd,
                stmbias=stmbias,
                tphysel=tphysel,
                tsmear=tsmear,
                usewvl=usewvl,
                nshiftk_orig=nshiftk_orig,
                nshiftk=nshiftk,
                mband=mband,
                istwfk=istwfk,
                istwfk_raw=istwfk.copy(),
                nband=nband,
                npwarr=npwarr,
                so_psp=so_psp,
                symafm=symafm,
                symrel=symrel,
                typat=typat,
                kpt=kpt,
                occ=occ,
                tnons=tnons,
                znucltypat=znucltypat,
                wtk=wtk,
                residm=residm,
                xred=xred,
                etot=etot,
                fermie=fermie,
                amu=amu,
                kptopt=kptopt,
                pawcpxocc=pawcpxocc,
                nelect=nelect,
                cellcharge=cellcharge,
                icoulomb=icoulomb,
                kptrlatt=kptrlatt,
                kptrlatt_orig=kptrlatt_orig,
                shiftk_orig=shiftk_orig,
                shiftk=shiftk,
                title=title,
                znuclpsp=np.array(znuclpsp),
                zionpsp=np.array(zionpsp),
                pspso=pspso,
                pspdat=pspdat,
                pspcod=pspcod,
                pspxc=pspxc,
                lmn_size=lmn_size,
                md5_pseudos=md5_pseudos,
                gmet=_gmet_from_rprimd(rprimd),
            )
    
            # Body: locate records without reading payloads (bounded scratch).
            index = {}
            formeig = None
            pos = fh.tell()
            rec_id = 0
            for spin in range(nsppol):
                for ik in range(nkpt):
                    npw_k = int(npwarr[ik])
                    nband_k = int(nband[ik + spin * nkpt])
                    offsets = {}
                    # REC_NPW
                    pos, n = _seek_record(fh, pos, rec_id, path)
                    offsets["npw"] = (pos, n)
                    if n != 12:
                        raise WfkReaderError(
                            "D5",
                            path,
                            f"record {rec_id}",
                            f"REC_NPW payload expected 12 bytes (npw, nspinor, "
                            f"nband_disk), found {n}",
                        )
                    fh.seek(pos)
                    counts = tuple(np.frombuffer(fh.read(12), dtype="<i4"))
                    expected_counts = (npw_k, nspinor, nband_k)
                    if counts != expected_counts:
                        _packed_spinor_guard(header, ik, path)
                        raise WfkReaderError(
                            "D2",
                            path,
                            f"REC_NPW[k={ik}, spin={spin}]",
                            f"stored(npw,nspinor,nband)={counts} != "
                            f"header(npwarr,nspinor,nband)={expected_counts}",
                        )
                    pos += n + 4
                    # REC_KG
                    pos, n = _seek_record(fh, pos, rec_id + 1, path)
                    if n != 3 * npw_k * 4:
                        raise WfkReaderError(
                            "D2",
                            path,
                            f"kg[k={ik}]",
                            f"kg record size {n} bytes implies "
                            f"{n // 12} G vectors, header npwarr({ik}) = {npw_k}",
                        )
                    offsets["kg"] = (pos, n)
                    pos += n + 4
                    # REC_EIG (+occ) — one record for formeig=0, per-band for 1
                    pos, n = _seek_record(fh, pos, rec_id + 2, path)
                    if formeig is None:
                        formeig = 0 if n == 2 * nband_k * 8 else 1
                    if formeig == 0:
                        if n != 2 * nband_k * 8:
                            raise WfkReaderError(
                                "D5",
                                path,
                                f"record {rec_id + 2}",
                                f"eigen+occ record expected {2 * nband_k * 8} bytes "
                                f"(nband={nband_k}), found {n}",
                            )
                        offsets["eig"] = (pos, nband_k * 8)
                        pos += n + 4
                        cg_recs = []
                        for band in range(nband_k):
                            pos, n = _seek_record(fh, pos, rec_id + 3 + band, path)
                            exp = 2 * npw_k * nspinor * 8
                            if n != exp:
                                _packed_spinor_guard(header, ik, path)
                                raise WfkReaderError(
                                    "D5",
                                    path,
                                    f"record {rec_id + 3 + band}",
                                    f"cg record expected {exp} bytes "
                                    f"(npw={npw_k}, nspinor={nspinor}), found {n}",
                                )
                            cg_recs.append((pos, n))
                            pos += n + 4
                        offsets["cg"] = cg_recs
                    else:
                        cg_recs = []
                        for band in range(nband_k):
                            if n != 2 * nband_k * 8:
                                raise WfkReaderError(
                                    "D5",
                                    path,
                                    f"record {rec_id + 3 + 2 * band}",
                                    f"eigen record expected {2 * nband_k * 8} bytes, "
                                    f"found {n}",
                                )
                            offsets.setdefault("eig", (pos, nband_k * 8))
                            pos += n + 4
                            pos, n = _seek_record(fh, pos, rec_id + 4 + 2 * band, path)
                            exp = 2 * npw_k * nspinor * 8
                            if n != exp:
                                raise WfkReaderError(
                                    "D5",
                                    path,
                                    f"record {rec_id + 4 + 2 * band}",
                                    f"cg record expected {exp} bytes, found {n}",
                                )
                            cg_recs.append((pos, n))
                            pos += n + 4
                        offsets["cg"] = cg_recs
                    index[(ik, spin)] = offsets
                    rec_id += 3 + nband_k * (1 if formeig == 0 else 2)
            header = _validate_header(header, path)
            return cls(path, "fortran-record", fh, header, index)
        except BaseException:
            fh.close()
            raise

    # -- NetCDF flavour ------------------------------------------------------

    @classmethod
    def _open_netcdf(cls, path):
        try:
            import netCDF4
        except ImportError as exc:
            # pragma: no cover
                raise WfkReaderError(
                    "D1",
                    path,
                    "netCDF4",
                    f"NetCDF WFK requires the netCDF4 python package: {exc}",
                ) from exc
        fh = netCDF4.Dataset(path, "r")
        try:
            fh.set_auto_mask(False)
            try:
                codvsn = b"".join(fh["codvsn"][:]).decode("ascii").strip("\x00 ")
            except (KeyError, IndexError) as exc:
                raise WfkReaderError(
                    "D1", path, "codvsn", f"missing codvsn variable: {exc}"
                ) from exc
            try:
                major = int(codvsn.split(".")[0])
            except ValueError:
                major = None
            if major not in SUPPORTED_ABINIT_MAJOR_VERSIONS:
                raise WfkReaderError(
                    "D1",
                    path,
                    "codvsn",
                    f"unsupported codvsn {codvsn!r}; supported major versions are "
                    f"{list(SUPPORTED_ABINIT_MAJOR_VERSIONS)}",
                )
    
            def need(name):
                if name not in fh.variables:
                    raise WfkReaderError(
                        "D7", path, name, f"required header variable {name!r} absent"
                    )
                return fh[name][:]
    
            nkpt = len(fh.dimensions["number_of_kpoints"])
            nsppol = len(fh.dimensions["number_of_spins"])
            nspinor = len(fh.dimensions["number_of_spinor_components"])
            nspden = len(fh.dimensions["number_of_components"])
            nband_ks = np.array(need("number_of_states")).ravel().astype(np.int32)
            bantot = int(nband_ks.sum())
            # NetCDF reverses Fortran axes: primitive_vectors already has rows.
            rprimd = np.array(need("primitive_vectors"), dtype=float)
            gmet = _gmet_from_rprimd(rprimd)
            istwfk = np.array(need("istwfk"), dtype=np.int32)
            header = WfkHeader(
                codvsn=codvsn,
                headform=int(need("headform")),
                fform=int(need("fform")),
                bantot=bantot,
                date=int(need("date")),
                intxc=int(need("intxc")),
                ixc=int(need("ixc")),
                natom=len(fh.dimensions["number_of_atoms"]),
                ngfft=np.array(
                    [
                        len(fh.dimensions[f"number_of_grid_points_vector{i}"])
                        for i in (1, 2, 3)
                    ]
                ),
                nkpt=nkpt,
                nspden=nspden,
                nspinor=nspinor,
                nsppol=nsppol,
                nsym=len(fh.dimensions["number_of_symmetry_operations"]),
                npsp=len(fh.dimensions["npsp"]),
                ntypat=len(fh.dimensions["number_of_atom_species"]),
                occopt=int(need("occopt")),
                pertcase=int(need("pertcase")),
                usepaw=int(need("usepaw")),
                ecut=float(np.array(need("kinetic_energy_cutoff")).ravel()[0]),
                ecutdg=float(np.array(fh["ecutdg"][:]).ravel()[0])
                if "ecutdg" in fh.variables
                else float(np.array(need("kinetic_energy_cutoff")).ravel()[0]),
                ecutsm=float(np.array(fh["ecutsm"][:]).ravel()[0])
                if "ecutsm" in fh.variables
                else 0.0,
                ecut_eff=float(np.array(need("ecut_eff")).ravel()[0]),
                qptn=np.array(fh["qptn"][:]).ravel()[:3].astype(float),
                rprimd=rprimd,
                stmbias=float(np.array(fh["stmbias"][:]).ravel()[0])
                if "stmbias" in fh.variables
                else 0.0,
                tphysel=float(np.array(fh["tphysel"][:]).ravel()[0])
                if "tphysel" in fh.variables
                else 0.0,
                tsmear=float(np.array(fh["tsmear"][:]).ravel()[0])
                if "tsmear" in fh.variables
                else 0.0,
                usewvl=int(need("usewvl")),
                nshiftk_orig=len(fh.dimensions["nshiftk_orig"]),
                nshiftk=len(fh.dimensions["nshiftk"]),
                mband=len(fh.dimensions["max_number_of_states"]),
                istwfk=istwfk,
                istwfk_raw=istwfk.copy(),
                nband=nband_ks,
                npwarr=np.array(need("number_of_coefficients")).ravel().astype(np.int32),
                so_psp=np.array(need("so_psp")).astype(np.int32),
                symafm=(
                    np.array(fh["symafm"][:]).astype(np.int32)
                    if "symafm" in fh.variables
                    else None
                ),
                symrel=(
                    np.array(fh["reduced_symmetry_matrices"][:]).astype(np.int32)
                    if "reduced_symmetry_matrices" in fh.variables
                    else None
                ),
                typat=np.array(need("atom_species")).astype(np.int32),
                kpt=np.array(need("reduced_coordinates_of_kpoints"), dtype=float),
                occ=np.array(need("occupations"), dtype=float),
                tnons=(
                    np.array(fh["reduced_symmetry_translations"][:], dtype=float)
                    if "reduced_symmetry_translations" in fh.variables
                    else None
                ),
                znucltypat=np.array(fh["atomic_numbers"][:]).astype(np.int32),
                wtk=np.array(need("kpoint_weights"), dtype=float),
                residm=float(np.array(fh["residm"][:]).ravel()[0]),
                xred=np.array(need("reduced_atom_positions"), dtype=float),
                etot=float(np.array(fh["etot"][:]).ravel()[0])
                if "etot" in fh.variables
                else float("nan"),
                fermie=float(np.array(fh["fermi_energy"][:]).ravel()[0])
                if "fermi_energy" in fh.variables
                else float("nan"),
                amu=np.array(fh["amu"][:], dtype=float)
                if "amu" in fh.variables
                else np.zeros(len(fh.dimensions["number_of_atom_species"])),
                kptopt=int(need("kptopt")),
                pawcpxocc=int(need("pawcpxocc")) if "pawcpxocc" in fh.variables else 0,
                nelect=float(np.array(need("number_of_electrons")).ravel()[0]),
                cellcharge=float(np.array(need("charge")).ravel()[0]),
                icoulomb=int(need("icoulomb")) if "icoulomb" in fh.variables else 0,
                kptrlatt=np.array(need("kptrlatt")).astype(np.int32),
                kptrlatt_orig=np.array(fh["kptrlatt_orig"][:]).astype(np.int32)
                if "kptrlatt_orig" in fh.variables
                else np.array(need("kptrlatt")).astype(np.int32),
                shiftk_orig=np.array(fh["shiftk_orig"][:], dtype=float)
                if "shiftk_orig" in fh.variables
                else np.array(need("shiftk"), dtype=float),
                shiftk=np.array(need("shiftk"), dtype=float),
                title=[
                    t.decode("ascii", "replace").strip("\x00 ")
                    if isinstance(t, bytes)
                    else str(t)
                    for t in (
                        b"".join(row).decode("ascii", "replace").strip("\x00 ")
                        for row in fh["title"][:]
                    )
                ]
                if "title" in fh.variables
                else ["" for _ in range(len(fh.dimensions["npsp"]))],
                znuclpsp=np.array(fh["znuclpsp"][:], dtype=float).ravel()
                if "znuclpsp" in fh.variables
                else np.zeros(len(fh.dimensions["npsp"])),
                zionpsp=np.array(fh["zionpsp"][:], dtype=float).ravel()
                if "zionpsp" in fh.variables
                else np.zeros(len(fh.dimensions["npsp"])),
                pspso=np.array(fh["pspso"][:]).astype(np.int32).ravel()
                if "pspso" in fh.variables
                else np.ones(len(fh.dimensions["npsp"]), np.int32),
                pspdat=np.array(fh["pspdat"][:]).astype(np.int32).ravel()
                if "pspdat" in fh.variables
                else np.zeros(len(fh.dimensions["npsp"]), np.int32),
                pspcod=np.array(fh["pspcod"][:]).astype(np.int32).ravel()
                if "pspcod" in fh.variables
                else np.zeros(len(fh.dimensions["npsp"]), np.int32),
                pspxc=np.array(fh["pspxc"][:]).astype(np.int32).ravel()
                if "pspxc" in fh.variables
                else np.zeros(len(fh.dimensions["npsp"]), np.int32),
                lmn_size=np.array(need("lmn_size")).astype(np.int32).ravel(),
                md5_pseudos=[
                    b"".join(row).decode("ascii").strip("\x00 ")
                    for row in np.atleast_2d(need("md5_pseudos"))
                ],
                gmet=gmet,
            )
            header = _validate_header(header, path)
            return cls(path, "netcdf", fh, header, None)
        except BaseException:
            fh.close()
            raise

    # -- public seam ---------------------------------------------------------

    @property
    def path(self):
        return str(self._path)

    @property
    def format(self):
        return self._fmt

    @property
    def header(self) -> WfkHeader:
        return self._header

    @property
    def peak_live_bytes(self) -> int:
        """Peak tracked numeric payload buffers; excludes Python/library overhead."""
        return self._peak_live_bytes

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internal bookkeeping -------------------------------------------------

    def _alloc(self, nbytes):
        self._live_bytes += int(nbytes)
        self._peak_live_bytes = max(self._peak_live_bytes, self._live_bytes)

    def _free(self, nbytes):
        self._live_bytes -= int(nbytes)

    def _check_ksb(self, k, spin, band_range):
        h = self._header
        if not (0 <= int(k) < h.nkpt):
            raise WfkReaderError(
                "D6",
                self._path,
                f"k index {int(k)}",
                f"k index out of range; header nkpt={h.nkpt}",
            )
        if not (0 <= int(spin) < h.nsppol):
            raise WfkReaderError(
                "D6",
                self._path,
                f"spin index {int(spin)}",
                f"spin index out of range; header nsppol={h.nsppol}",
            )
        nb = int(h.nband[int(k) + int(spin) * h.nkpt])
        b0, b1 = band_range
        if not (0 <= b0 < b1 <= nb):
            raise WfkReaderError(
                "D6",
                self._path,
                f"(band-range=({b0},{b1}), k={int(k)}, spin={int(spin)})",
                f"requested band range outside header nband({int(k)},{int(spin)})={nb}",
            )
        return nb

    def _check_packed_spinor(self, ik):
        """D3: packed storage is forbidden for spinor / spin-orbit-PAW states."""
        h = self._header
        mode = self._resolved_istwfk(ik)
        so = int(np.max(np.asarray(h.so_psp, dtype=int), initial=0))
        if mode > 1 and (h.nspinor != 1 or h.pawspnorb or so > 1):
            raise WfkReaderError(
                "D3",
                self._path,
                f"k index {int(ik)}",
                f"packed storage resolved istwfk={mode} requested but forbidden: "
                f"nspinor={h.nspinor}, pawspnorb={h.pawspnorb} "
                f"(native restriction {_NATIVE_PACKED_RESTRICTION}: istwfk is "
                "forced to 1 for nspinor!=1 or pawspnorb!=0)",
            )

    def _resolved_istwfk(self, ik):
        h = self._header
        raw = int(h.istwfk_raw[ik])
        resolved = resolve_istwfk(
            raw, h.kpt[ik], nspinor=h.nspinor, pawspnorb=h.pawspnorb
        )
        if int(h.istwfk[ik]) != resolved:
            resolved = int(h.istwfk[ik])
            # stored resolved value must be a legal mode consistent with kpt
            if not (1 <= resolved <= 9):
                raise WfkReaderError(
                    "D3",
                    self._path,
                    f"istwfk[k={int(ik)}]",
                    f"stored istwfk value {resolved} at k index {int(ik)} is "
                    f"outside the legal 1-9 (raw value {raw})",
                )
            if resolved > 1:
                expected = resolve_istwfk(0, h.kpt[ik])
                if expected != resolved:
                    raise WfkReaderError(
                        "D3",
                        self._path,
                        f"istwfk[k={int(ik)}]",
                        f"stored istwfk {resolved} inconsistent with k-point "
                        f"{list(h.kpt[ik])} which resolves to {expected} "
                        f"(raw value {raw})",
                    )
        return resolved

    # -- eigenvalue slabs (AC-060-06) -----------------------------------------

    def eigenvalues(self, k, spin, band_range):
        """Return eigenvalues exactly as stored (Hartree), no conversion.

        Reads only the eigenvalue record/slab; wavefunction records are not
        touched and no second band filter is applied.
        """
        self._check_ksb(k, spin, band_range)
        b0, b1 = band_range
        if self._fmt == "netcdf":
            if "eigenvalues" not in self._fh.variables:
                raise WfkReaderError(
                    "D6",
                    self._path,
                    f"eigenvalues[k={k}, spin={spin}, bands={band_range}]",
                    f"missing eigenvalues variable; header nband={self._header.nband[k+spin*self._header.nkpt]}",
                )
            var = self._fh.variables["eigenvalues"]
            self._alloc((b1 - b0) * 8)
            try:
                out = np.asarray(var[spin, k, b0:b1], dtype=np.float64)
                if out.shape != (b1 - b0,) or not np.all(np.isfinite(out)):
                    raise WfkReaderError(
                        "D6",
                        self._path,
                        f"eigenvalues[k={k}, spin={spin}, bands={band_range}]",
                        f"complete finite slab required; shape={out.shape}, header nband="
                        f"{self._header.nband[k+spin*self._header.nkpt]}",
                    )
            except (IndexError, KeyError) as exc:
                raise WfkReaderError(
                    "D6",
                    self._path,
                    f"eigenvalues[k={k}, spin={spin}, bands={band_range}]",
                    f"missing requested slab; header nband={self._header.nband[k+spin*self._header.nkpt]}",
                ) from exc
            finally:
                self._free((b1 - b0) * 8)
            return out
        pos, n = self._index[(int(k), int(spin))]["eig"]
        self._fh.seek(pos)
        buf = self._fh.read(n)
        if len(buf) != n:
            raise WfkReaderError(
                "D5",
                self._path,
                f"eigen record (k={int(k)}, spin={int(spin)})",
                f"expected {n} bytes, found {len(buf)}",
            )
        slab_bytes = (b1 - b0) * 8
        self._alloc(n + slab_bytes)
        try:
            return np.frombuffer(buf, "<f8")[b0:b1].copy()
        finally:
            self._free(n + slab_bytes)

    # -- stored coefficient blocks (AC-060-01/03/04) ---------------------------

    def kg(self, k):
        """Stored G vectors (3, npw_k) for one k-point, exactly as stored.

        Fails closed (D2) when the stored kg count disagrees with the header
        ``npwarr`` or a stored G falls outside the G-sphere bound.
        """
        h = self._header
        ik = int(k)
        if not (0 <= ik < h.nkpt):
            raise WfkReaderError(
                "D6",
                self._path,
                f"k index {ik}",
                f"k index out of range; header nkpt={h.nkpt}",
            )
        npw = int(h.npwarr[ik])
        if self._fmt == "netcdf":
            var = self._fh["reduced_coordinates_of_plane_waves"]
            self._alloc(npw * 12)
            try:
                kg = np.array(var[ik, :npw, :], dtype=np.int32).T.copy()
            finally:
                self._free(npw * 12)
        else:
            pos, n = self._index[(ik, 0)]["kg"]
            self._fh.seek(pos)
            buf = self._fh.read(n)
            self._alloc(n)
            try:
                # kg_k(3, npw) Fortran ravel = interleaved (x,y,z) triples
                kg = np.frombuffer(buf, "<i4").reshape(npw, 3, order="C").T.copy()
            finally:
                self._free(n)
        if kg.shape[1] != npw:
            raise WfkReaderError(
                "D2",
                self._path,
                f"kg[k={ik}]",
                f"kg has {kg.shape[1]} G vectors but header npwarr({ik}) = {npw}",
            )
        if not _kg_within_sphere(h.kpt[ik], kg, h.gmet, h.ecut_eff):
            raise WfkReaderError(
                "D2",
                self._path,
                f"kg[k={ik}]",
                f"a stored G vector at k index {ik} falls outside the G-sphere "
                f"bound ecut_eff={h.ecut_eff} Hartree",
            )
        return kg

    def coefficients(self, k, spin, band_range):
        """Stored coefficient block (nband, npw, nspinor) complex128.

        Packed storage is returned exactly as stored — no weights applied.
        """
        self._check_ksb(k, spin, band_range)
        self._check_packed_spinor(k)
        h = self._header
        b0, b1 = band_range
        npw = int(h.npwarr[k])
        if self._fmt == "netcdf":
            var = self._fh["coefficients_of_wavefunctions"]
            self._alloc((b1 - b0) * npw * h.nspinor * 16)
            try:
                slab = np.asarray(var[spin, k, b0:b1, :, :npw], dtype=np.float64)
            finally:
                self._free((b1 - b0) * npw * h.nspinor * 16)
            # NetCDF real/imag axis is contiguous; borrow a complex view.
            return slab.view(np.complex128)[..., 0].transpose(0, 2, 1)
        recs = self._index[(int(k), int(spin))]["cg"]
        out = np.empty((b1 - b0, npw, h.nspinor), dtype=np.complex128)
        nbytes = out.nbytes
        self._alloc(nbytes)
        try:
            for ib, band in enumerate(range(b0, b1)):
                pos, n = recs[band]
                self._fh.seek(pos)
                self._alloc(n)
                try:
                    buf = self._fh.read(n)
                    if len(buf) != n:
                        raise WfkReaderError(
                            "D5",
                            self._path,
                            f"cg record (k={int(k)}, spin={int(spin)}, band={band})",
                            f"expected {n} bytes, found {len(buf)}",
                        )
                    # ABINIT cg(2,npw*nspinor): real/imag interleaved, G fastest.
                    vals = np.frombuffer(buf, "<c16")
                    out[ib] = vals.reshape(npw, h.nspinor, order="F")
                    del vals, buf
                finally:
                    self._free(n)
            return out
        finally:
            self._free(nbytes)

    # -- packed decode (AC-060-02) ---------------------------------------------

    def decode_full_sphere(self, k, spin, band_range):
        """Decode packed storage into full-sphere coefficients.

        Returns ``(kg_full (3,N) int, weights (N,) float, cg_full
        (nband, N, nspinor) complex)``.  The packed half-sphere storage
        weights are applied exactly once inside the reconstruction, so the
        returned ``weights`` are unit weights and ``sum(|cg_full|^2)`` is the
        full-storage norm.
        """
        self._check_ksb(k, spin, band_range)
        self._check_packed_spinor(k)
        h = self._header
        ik = int(k)
        mode = self._resolved_istwfk(ik)
        kg_stored = self.kg(ik)
        cg_stored = self.coefficients(k, spin, band_range)
        return decode_packed_sphere(
            h, ik, mode, kg_stored, cg_stored, band_range, self._path
        )

    # -- symmetry metadata (FR-020) ---------------------------------------------

    def require_symmetry(self):
        """Return (symrel, tnons, symafm); D7 when absent but requested."""
        h = self._header
        if self._fmt == "netcdf":
            for name, value in (
                ("reduced_symmetry_matrices", h.symrel),
                ("reduced_symmetry_translations", h.tnons),
                ("symafm", h.symafm),
            ):
                if name not in self._fh.variables or value is None or value.size == 0:
                    raise WfkReaderError(
                        "D7",
                        self._path,
                        name,
                        f"required symmetry metadata absent; expected native header variable {name!r}",
                    )
        return h.symrel, h.tnons, h.symafm


def decode_packed_sphere(header, ik, mode, kg_stored, cg_stored, band_range, path):
    """Shared packed-storage decode (epic-048-c certified conventions).

    Used identically by the WFK reader and the caller-array source, so both
    carriers expose one decode path.  Returns ``(kg_full (3,N) int, weights
    (N,) float, cg_full (nband, N, nspinor) complex)`` with unit weights —
    the packed half-sphere storage weights are applied exactly once inside
    the reconstruction.
    """
    h = header
    npw = kg_stored.shape[1]
    b0, b1 = band_range
    if mode == 1:
        return (
            kg_stored,
            np.ones(npw, dtype=np.float64),
            cg_stored,
        )
    g0 = np.round(2.0 * h.kpt[ik]).astype(np.int64)
    if not np.allclose(2.0 * h.kpt[ik], g0, atol=_TOL10):
        raise WfkReaderError(
            "D3",
            path,
            f"istwfk[k={ik}]",
            f"packed mode {mode} requires k-point on the 2k=G0 integer "
            f"grid; got kpt={list(h.kpt[ik])}",
        )
    index = {tuple(g): i for i, g in enumerate(kg_stored.T)}
    # verify each stored G has at most one distinct partner
    order = list(index.items())
    new_gs = []
    new_set = set()
    for g, i in order:
        p = tuple(-np.asarray(g) - g0)
        if p in index:
            if p == g:
                continue
            raise WfkReaderError(
                "D2",
                path,
                f"kg[k={ik}]",
                f"packed storage contains both G={list(g)} and its "
                f"involution partner {list(p)} (npw={npw}); the packed "
                "half-sphere must store one representative per pair",
            )
        if p not in new_set and p != g:
            new_set.add(p)
            new_gs.append(np.array(p, dtype=np.int32))
    n_new = len(new_gs)
    n_full = npw + n_new
    kg_full = np.empty((3, n_full), dtype=np.int32)
    kg_full[:, :npw] = kg_stored
    if n_new:
        kg_full[:, npw:] = np.array(new_gs).T
    # The packed half-sphere storage weights (2 per stored plane wave with
    # a distinct conjugate partner, 1 at Gamma G=0 and for self-conjugate
    # planes) are applied exactly once, inside the reconstruction: the
    # returned full sphere carries unit norm weights, so
    # sum(|cg_full|^2) equals the full-storage norm.  Applying the packed
    # weights again on the returned sphere double-counts and is detected
    # by the internal consistency check below.
    weights = np.ones(n_full, dtype=np.float64)
    cg_full = np.empty((b1 - b0, n_full, h.nspinor), dtype=np.complex128)
    cg_full[:, :npw, :] = cg_stored
    new_index = {tuple(g): k for k, g in enumerate(new_gs)}
    for i, g in enumerate(kg_full.T[:npw]):
        p = tuple(-np.asarray(g) - g0)
        if p != tuple(g):
            j = npw + new_index[p]
            cg_full[:, j, :] = np.conj(cg_stored[:, i, :])
    # internal consistency: ABINIT packed weights on the stored half must
    # reproduce the full-sphere norm
    packed_w = np.where(
        np.array(
            [tuple(g) != tuple(-np.asarray(g) - g0) for g in kg_stored.T], dtype=float
        ),
        2.0,
        1.0,
    )
    norm_full = float(np.sum(np.abs(cg_full[:, :, 0]) ** 2, axis=1).sum())
    norm_packed = float(
        np.sum((packed_w[None, :] * np.abs(cg_stored[:, :, 0]) ** 2).sum())
    )
    if abs(norm_full - norm_packed) > 1e-9 * max(1.0, norm_full):
        raise WfkReaderError(
            "D2",
            path,
            f"kg[k={ik}]",
            f"packed-storage norm check failed: packed weights give "
            f"{norm_packed}, reconstructed full sphere gives {norm_full}",
        )
    return kg_full, weights, cg_full


def _packed_spinor_guard(header, ik, path):
    """D3: a packed record whose size implies forbidden spinor/s.o. storage."""
    so = int(np.max(np.asarray(header.so_psp, dtype=int), initial=0))
    mode = int(header.istwfk[ik])
    if mode > 1 and (header.nspinor != 1 or header.pawspnorb or so > 1):
        raise WfkReaderError(
            "D3",
            path,
            f"k index {ik}",
            f"packed storage resolved istwfk={mode} encountered but forbidden: "
            f"nspinor={header.nspinor}, pawspnorb={header.pawspnorb} "
            f"(native restriction m_inkpts.F90:501-506: istwfk is forced to 1 "
            "for nspinor!=1 or pawspnorb!=0)",
        )


def _seek_record(fh, pos, rec_id, path):
    fh.seek(pos)
    head = fh.read(4)
    if len(head) != 4:
        raise WfkReaderError(
            "D5",
            path,
            f"record {rec_id}",
            f"record marker truncated at byte {pos}: expected 4, found {len(head)}",
        )
    (n,) = struct.unpack("<I", head)
    fh.seek(pos + 4 + n)
    tail = fh.read(4)
    if len(tail) != 4 or struct.unpack("<I", tail)[0] != n:
        actual = (
            struct.unpack("<I", tail)[0] if len(tail) == 4 else f"{len(tail)} bytes"
        )
        raise WfkReaderError(
            "D5",
            path,
            f"record {rec_id}",
            f"trailing marker expected{n}, found{actual} at byte{pos+4+n}",
        )
    return pos + 4, n


def _validate_header(h, path):
    resolved = np.empty(h.nkpt, dtype=np.int32)
    for ik in range(h.nkpt):
        raw = int(h.istwfk_raw[ik])
        try:
            mode = resolve_istwfk(
                raw, h.kpt[ik], nspinor=h.nspinor, pawspnorb=h.pawspnorb
            )
        except WfkReaderError as error:
            raise WfkReaderError(
                error.diagnostic, path, f"istwfk[k={ik}]", error.detail
            ) from None
        resolved[ik] = mode
    # Preserve the raw record independently; consumers always see resolved modes.
    h = replace(h, istwfk=resolved, title=tuple(h.title), md5_pseudos=tuple(h.md5_pseudos))
    for item in fields(h):
        value = getattr(h, item.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return h


def _kg_within_sphere(kpt, kg, gmet, ecut_eff):
    k = np.asarray(kpt, dtype=float)
    d = k[None, :] + kg.T.astype(float)
    return np.all(0.5 * np.einsum("ij,jk,ik->i", d, gmet, d) <= ecut_eff * (1 + 1e-8))


def cross_format_check(path_a, path_b):
    """D8: verify two WFK files (binary and/or NetCDF) describe the same deck.

    Raises a D8 diagnostic naming the first mismatching header field.
    """
    ra = WfkReader.open(path_a)
    try:
        rb = WfkReader.open(path_b)
        try:
            ha, hb = ra.header, rb.header
            scalars = [
                "nkpt",
                "nsppol",
                "nspinor",
                "nspden",
                "natom",
                "nsym",
                "npsp",
                "ntypat",
                "mband",
                "bantot",
                "usepaw",
                "occopt",
                "kptopt",
                "nelect",
                "headform",
            ]
            for name in scalars:
                va, vb = getattr(ha, name), getattr(hb, name)
                if va != vb:
                    raise WfkReaderError(
                        "D8",
                        f"{path_a} vs {path_b}",
                        name,
                        f"cross-format disagreement: {va!r} != {vb!r}",
                    )
            arrays = {
                "kpt": (ha.kpt, hb.kpt, 1e-12),
                "npwarr": (ha.npwarr, hb.npwarr, 0),
                "istwfk": (ha.istwfk, hb.istwfk, 0),
                "nband": (ha.nband, hb.nband, 0),
                "rprimd": (ha.rprimd, hb.rprimd, 1e-12),
                "typat": (ha.typat, hb.typat, 0),
            }
            for name, (va, vb, tol) in arrays.items():
                if va.shape != vb.shape or not np.allclose(
                    va, vb, rtol=0, atol=tol + 1e-12
                ):
                    raise WfkReaderError(
                        "D8",
                        f"{path_a} vs {path_b}",
                        name,
                        f"cross-format disagreement: shapes {va.shape} vs "
                        f"{vb.shape} or values differ beyond {tol}",
                    )
        finally:
            rb.close()
    finally:
        ra.close()
