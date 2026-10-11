"""Characterize, freeze, generate and consume; run external executables yourself."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

import numpy as np

from prepare import checked_datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lawaf.interfaces.abinit.basis_datasets import (
    read_paw_xml_dataset,
    read_upf_projectors,
)
from lawaf.interfaces.abinit.nc_trials import NcTrialBasis
from lawaf.interfaces.abinit.paw_kernels import HARMONIC_CONVENTION
from lawaf.interfaces.abinit.paw_provider import PawCoefficientProvider
from lawaf.interfaces.abinit.paw_trials import PawTrialBasis
from lawaf.interfaces.abinit.physics_paw import PawOverlap
from lawaf.interfaces.abinit.producer import (
    AbinitWannierInputs,
    GenerationPlan,
    HA_EV,
    parse_nnkp,
)
from lawaf.interfaces.abinit.source_wfk import WfkReader
from lawaf.interfaces.abinit.trial_manifest import default_nc_manifest, parse_manifest


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trial_span_character(overlaps, gram):
    """Normalized-state weight in a nonorthogonal trial span, in its true metric."""
    if not np.isfinite(overlaps).all() or not np.isfinite(gram).all():
        raise ValueError("D3 non-finite trial-span overlap/metric")
    if np.max(abs(gram - gram.conj().T)) > 1e-12 or np.linalg.eigvalsh(gram).min() <= 0:
        raise ValueError("D3 trial-span metric must be Hermitian and positive definite")
    return np.einsum(
        "bi,ib->b", overlaps, np.linalg.solve(gram, overlaps.conj().T)
    ).real


def gamma_sp_character(source, basis_path, provider):
    indices = np.flatnonzero(
        np.all(source.header.kpt == np.rint(source.header.kpt), axis=1)
    )
    if len(indices) != 1:
        raise ValueError(
            "D3 source mesh must contain exactly one Gamma point for near-edge characterization"
        )
    gamma = int(indices[0])
    manifest = parse_manifest(basis_path)
    basis = (
        PawTrialBasis(manifest, provider)
        if provider is not None
        else NcTrialBasis(manifest, source.header)
    )
    kg, _, states = source.decode_full_sphere(gamma, 0, (0, 24))
    trials = basis(gamma, kg.T)
    overlaps = states[:, :, 0].conj() @ trials
    gram = trials.conj().T @ trials
    if provider is not None:
        metric = PawOverlap(provider)
        dual = basis.dual_coefficients(gamma)[:, :, None]
        overlaps += metric.projection(provider.band_block(gamma, 0, (0, 24)), dual)
        gram += metric.projection(dual.transpose(1, 2, 0), dual)
    characters = trial_span_character(overlaps, gram)
    if not characters[10:18].min() > characters[18:24].max():
        raise ValueError(
            f"D3 Gamma 4s/4p trial-span characters do not separate declared eight near-edge states: {characters.tolist()}"
        )
    return gamma, characters


def write_basis(folder, pspdir, kind, header, semicore=False):
    path = folder / ("semicore-basis.json" if semicore else "basis.json")
    sites = [
        {
            "label": name,
            "species": name,
            "position": list(position),
            "local_axes": np.eye(3).tolist(),
        }
        for name, position in zip(("Ga", "As"), header.xred)
    ]
    if kind == "nc" and not semicore:
        return default_nc_manifest(
            path,
            [
                (name, pspdir / "Psdj_nc_fr_04_pbe_std_psp8" / f"{name}.upf")
                for name in ("Ga", "As")
            ],
            sites,
        )
    species = []
    for name in ("Ga", "As"):
        if kind == "nc":
            dataset_path = pspdir / "Psdj_nc_fr_04_pbe_std_psp8" / f"{name}.upf"
            dataset = read_upf_projectors(dataset_path)
            channel = dataset.get_channel("PP_CHI.1")
            if channel.l != 2 or channel.label.upper() != "3D":
                raise ValueError(
                    f"D3 {dataset_path}: PP_CHI.1 must be actual3d radial reference"
                )
            radials = [
                {
                    "id": channel.radial_id,
                    "label": channel.label,
                    "l": 2,
                    "harmonics": list(range(5)),
                }
            ]
        else:
            dataset_path = pspdir / "Psdj_paw_pbe_std" / f"{name}.xml"
            dataset = read_paw_xml_dataset(dataset_path)
            selected = []
            for n, l in ((3, 2),) if semicore else ((4, 0), (4, 1)):
                candidates = [
                    channel
                    for channel in dataset.channels
                    if channel.n == n and channel.l == l and channel.occupation > 0
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"D3 {dataset_path}: unambiguous occupied n{n}/l{l} radial reference required"
                    )
                selected.append(candidates[0])
            radials = [
                {
                    "id": channel.state_id,
                    "label": f"{channel.n}{'spdf'[channel.l]}",
                    "l": channel.l,
                    "harmonics": list(range(2 * channel.l + 1)),
                }
                for channel in selected
            ]
        species.append(
            {
                "name": name,
                "kind": "nc_upf" if kind == "nc" else "paw_xml",
                "dataset_path": str(dataset_path),
                "sha256": digest(dataset_path),
                "radial_ids": radials,
            }
        )
    document = {
        "schema": "lawaf_trial_basis_manifest",
        "version": "1.0",
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": HARMONIC_CONVENTION,
            "normalization": "normalized",
        },
        "trial_definition": "pseudo_radial"
        if kind == "nc"
        else "paw_augmented_partial_wave",
        "spin_frame_labels": False,
        "species": species,
        "sites": sites,
    }
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def require_source(folder, source_plan):
    prefix = f"gaas-{source_plan['kind']}"
    # A reused source may split its convergence evidence across the original
    # SCF report and a completion remainder run; every declared report is
    # required, and together they must show both residual and completion.
    names = source_plan.get("source_stdouts") or [f"{prefix}.stdout"]
    texts = []
    for name in names:
        output = folder / name
        if not output.is_file():
            raise ValueError(f"D2 {output}: source convergence report absent")
        text = output.read_text()
        if "!ScfConvergenceWarning" in text or "was not enough SCF cycles" in text:
            raise ValueError(f"D2 {output}: source run reported a convergence failure")
        texts.append(text)
    if not any("Calculation completed." in text for text in texts):
        raise ValueError(f"D2 {folder}: source run missing or nonconverged")
    converged = [
        line
        for text in texts
        for line in text.splitlines()
        if "vres2" in line and "=>converged" in line
    ]
    if not converged:
        raise ValueError(
            f"D2 {output}: explicit potential-residual convergence evidence missing"
        )
    residual = float(
        converged[-1]
        .split("vres2", 1)[1]
        .split("=", 1)[1]
        .split("<", 1)[0]
        .strip()
        .replace("D", "E")
    )
    if not 0 <= residual <= source_plan["scf_residual_target"]:
        raise ValueError(
            f"D2 {output}: actual SCF residual{residual} above declared target"
        )
    path = folder / f"{prefix}o_DS2_WFK.nc"
    if not path.is_file():
        raise ValueError(f"D2 {path}: full-mesh source absent")
    return path


def characterize(folder, pspdir):
    frozen = folder / "frozen-workflow.json"
    if frozen.exists() or (folder / "gaas.wout").exists():
        raise ValueError(
            f"D3 {folder}: freeze must precede localization; choose a fresh run folder"
        )
    source_plan = json.loads((folder / "source-plan.json").read_text())
    kind = source_plan["kind"]
    hashes = checked_datasets(pspdir, kind)
    path = require_source(folder, source_plan)
    with WfkReader.open(path) as source:
        h = source.header
        if (h.nsppol, h.nspinor, h.nspden) != (1, 1, 1) or h.mband != 24:
            raise ValueError(f"D2 {path}: required scalar24-physical-band source")
        if not 0 <= h.residm <= source_plan["nscf_residual_target"]:
            raise ValueError(
                f"D2 {path}: wave residual{h.residm} above declared target"
            )
        energies = (
            np.stack([source.eigenvalues(k, 0, (0, 24)) for k in range(h.nkpt)]) * HA_EV
        )
        core_top = float(energies[:, 9].max())
        valence_bottom = float(energies[:, 10].min())
        valence_top = float(energies[:, 13].max())
        conduction_bottom = float(energies[:, 14].min())
        if not core_top < valence_bottom or not valence_top < conduction_bottom:
            raise ValueError(
                f"D3 {path}: ambiguous semicore/valence or valence/conduction window; energies{core_top,valence_bottom,valence_top,conduction_bottom}"
            )
        basis_path = write_basis(folder, pspdir, kind, h)
        core_path = write_basis(folder, pspdir, kind, h, semicore=True)
        manifest = parse_manifest(core_path)
        provider = (
            PawCoefficientProvider(
                source,
                [pspdir / "Psdj_paw_pbe_std" / f"{name}.xml" for name in ("Ga", "As")],
            )
            if kind == "paw"
            else None
        )
        basis = (
            PawTrialBasis(manifest, provider)
            if provider is not None
            else NcTrialBasis(manifest, h)
        )
        metric = PawOverlap(provider) if provider is not None else None
        occupations = np.asarray(h.occ).reshape(h.nkpt, 24)
        if not np.all(occupations[:, :14] > 1.99) or not np.all(
            occupations[:, 14:] < 0.01
        ):
            raise ValueError(
                f"D3 {path}: four occupied retained valence states not certified by actual occupations"
            )
        character = []
        for k in range(h.nkpt):
            kg, _, states = source.decode_full_sphere(k, 0, (0, 24))
            trial = basis(k, kg.T)
            overlaps = states[:, :, 0].conj() @ trial
            norms = np.sum(abs(trial) ** 2, axis=0)
            if provider is not None:
                dual = basis.dual_coefficients(k)[:, :, None]
                overlaps += metric.projection(provider.band_block(k, 0, (0, 24)), dual)
                norms += np.diag(metric.projection(dual.transpose(1, 2, 0), dual)).real
            character.append(np.sum(abs(overlaps) ** 2 / norms[None], axis=1))
        character = np.asarray(character)
        core_min = float(character[:, :10].min())
        retained_max = float(character[:, 10:].max())
        if not core_min > retained_max:
            raise ValueError(
                f"D3 {path}: actual3d overlap characters do not separate first10 from retained14: {core_min,retained_max}"
            )
        lower = (core_top + valence_bottom) / 2
        upper = float(energies[:, 23].max()) + 0.5
        gamma, sp_characters = gamma_sp_character(source, basis_path, provider)
        gamma_near_edge_top = float(energies[gamma, 17])
        next_manifold_floor = float(energies[:, 18].min())
        if not gamma_near_edge_top < next_manifold_floor:
            raise ValueError(
                f"D3 no source-only frozen-window bracket for Gamma eight-state manifold: {gamma_near_edge_top,next_manifold_floor}"
            )
        frozen_upper = (gamma_near_edge_top + next_manifold_floor) / 2
        controls = {
            "dis_win_min": repr(lower),
            "dis_win_max": repr(upper),
            "dis_froz_min": repr(lower),
            "dis_froz_max": repr(frozen_upper),
            "num_iter": "1000",
            "conv_window": "5",
            "conv_tol": "1e-10",
            "dis_num_iter": "1000",
            "dis_conv_window": "5",
            "dis_conv_tol": "1e-10",
            "write_hr": "true",
            "write_xyz": "true",
            "fermi_energy": repr(float(h.fermie) * HA_EV),
        }
        validate_windows(energies, controls, gamma)
        record = {
            **source_plan,
            "dataset_sha256": hashes,
            "source_wfk": str(path),
            "source_wfk_sha256": digest(path),
            "basis_manifest": str(basis_path),
            "basis_manifest_sha256": digest(basis_path),
            "window_controls": controls,
            "characterization_policy": "source_only_gamma_sp_near_edge_v2",
            "gamma_source_index": gamma,
            "gamma_4sp_trial_span_characters": sp_characters.tolist(),
            "gamma_near_edge_top_eV": gamma_near_edge_top,
            "next_manifold_floor_eV": next_manifold_floor,
            "semicore_eV_max": core_top,
            "valence_eV_min": valence_bottom,
            "valence_eV_max": valence_top,
            "conduction_eV_min": conduction_bottom,
            "semicore3d_character_min": core_min,
            "retained3d_character_max": retained_max,
            "orbital_character_scores": character.tolist(),
            "manifold_certification": "semicore and Gamma valence/conduction trial-span separation certified from source-only data; cutoff and held-out interpolation remain independent gates",
        }
    frozen.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "semicore_eV_max",
                    "valence_eV_min",
                    "valence_eV_max",
                    "conduction_eV_min",
                    "semicore3d_character_min",
                    "retained3d_character_max",
                    "window_controls",
                )
            },
            indent=2,
        )
    )


def validate_windows(energies, controls, gamma_index):
    values = np.asarray(energies, dtype=float)
    lower, upper, frozen_lower, frozen_upper = (
        float(controls[name])
        for name in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max")
    )
    if (
        not np.isfinite(values).all()
        or not np.isfinite([lower, upper, frozen_lower, frozen_upper]).all()
    ):
        raise ValueError("D3 non-finite source/window values")
    if not lower <= frozen_lower < frozen_upper <= upper:
        raise ValueError("D3 empty or inconsistent outer/frozen window bounds")
    if lower > values[:, 10:24].min() or upper < values[:, 10:24].max():
        raise ValueError("D3 outer window excludes the characterized retained manifold")
    if (
        frozen_lower > values[:, 10:14].min()
        or frozen_upper < values[:, 10:14].max()
        or frozen_upper < values[gamma_index, 17]
    ):
        raise ValueError(
            "D3 frozen window excludes characterized valence/Gamma near-edge states"
        )
    retained = values[:, 10:24]
    frozen_count = np.sum(
        (retained >= frozen_lower) & (retained <= frozen_upper), axis=1
    )
    if np.any(frozen_count > 8):
        raise ValueError(
            f"D3 frozen window contains more than eight states: {frozen_count.tolist()}"
        )


def augmentation_witness(source, basis_path, nnkp_path):
    manifest = parse_manifest(basis_path)
    provider = PawCoefficientProvider(
        source, [manifest.dataset_paths[name] for name in ("Ga", "As")]
    )
    basis = PawTrialBasis(manifest, provider)
    metric = PawOverlap(provider)
    table = parse_nnkp(nnkp_path).nnkpts
    metric.prepare_neighbors(table)
    kg, _, states = source.decode_full_sphere(0, 0, (10, 24))
    electronic = provider.band_block(0, 0, (10, 24))
    trial = basis.dual_coefficients(0)[:, :, None]
    delta_a = metric.projection(electronic, trial)
    row = table[0, 0]
    shift = source.header.kpt[row[0]] + row[1:] - source.header.kpt[0]
    delta_m = metric.overlap(
        electronic, provider.band_block(int(row[0]), 0, (10, 24)), shift
    )
    pseudo = states[:, :, 0].conj() @ states[:, :, 0].T
    complete = pseudo + metric.projection(electronic, electronic.transpose(2, 0, 1))
    metric_error = float(np.max(abs(complete - np.eye(14))))
    removed_error = float(np.max(abs(pseudo - np.eye(14))))
    if metric_error > 1e-8 or removed_error <= 1e-3:
        raise ValueError(
            f"PAW native metric/augmentation-removal witness fails: {metric_error,removed_error}"
        )
    return {
        "A_augmentation_max": float(abs(delta_a).max()),
        "true_neighbor_M_augmentation_max": float(abs(delta_m).max()),
        "native_metric_error": metric_error,
        "augmentation_removed_metric_error": removed_error,
    }


def generate(folder, pspdir, matrices):
    frozen = json.loads((folder / "frozen-workflow.json").read_text())
    checked_datasets(pspdir, frozen["kind"])
    if (
        digest(frozen["source_wfk"]) != frozen["source_wfk_sha256"]
        or digest(frozen["basis_manifest"]) != frozen["basis_manifest_sha256"]
    ):
        raise ValueError("D1 source or characterized basis changed after window freeze")
    start = time.perf_counter()
    tracemalloc.start()
    try:
        with WfkReader.open(frozen["source_wfk"]) as source:
            energies = (
                np.stack(
                    [
                        source.eigenvalues(k, 0, (0, 24))
                        for k in range(source.header.nkpt)
                    ]
                )
                * HA_EV
            )
            validate_windows(
                energies, frozen["window_controls"], frozen["gamma_source_index"]
            )
            plan = GenerationPlan(
                tuple(frozen["retained_original_bands"]),
                frozen["num_wann"],
                tuple(frozen["mesh"]),
                controls=frozen["window_controls"],
                projections=("Ga:s;p", "As:s;p"),
            )
            producer = AbinitWannierInputs(
                source, plan, trial_manifest=frozen["basis_manifest"]
            )
            if not matrices:
                print(producer.write_win(folder, "gaas"))
                return
            result = producer.write_matrices(folder / "gaas.nnkp", folder, "gaas")
            augmentation = (
                augmentation_witness(
                    source, frozen["basis_manifest"], folder / "gaas.nnkp"
                )
                if frozen["kind"] == "paw"
                else None
            )
            _, peak = tracemalloc.get_traced_memory()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if peak > 256 * 1024**2 or rss > 2 * 1024**3:
                raise ValueError(
                    f"D6 producer RSS/scratch{rss,peak} exceed Architecture8 targets"
                )
            report = {
                "disk_bytes": result.bytes,
                "generation_io_seconds": result.seconds,
                "process_seconds": time.perf_counter() - start,
                "producer_peak_rss_bytes": rss,
                "producer_traced_peak_bytes": peak,
                "provenance": result.provenance,
            }
            if augmentation is not None:
                report["augmentation_witness"] = augmentation
            (folder / "producer-measurements.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(report, indent=2))
    finally:
        tracemalloc.stop()


def hamiltonio_provenance(wannier_ham_class):
    """Declare and enforce the supported HamiltonIO reader provenance.

    The story067 pinned runs use ``use_ws_distance=.true.``, whose Fourier
    interpolation needs the wsvec-aware ``_gen_ham_scheme2`` path (ADR-002, no
    ``val/2`` rescaling). Default-env HamiltonIO installs silently destroy the
    imaginary hopping parts and yield wrong held-out numbers (see
    ``.tmp/story067-spectral-diagnosis/REPORT.md``); a wrong reader must fail
    loudly instead of silently degrading the gate.
    """
    module = inspect.getmodule(wannier_ham_class)
    source = inspect.getsource(wannier_ham_class)
    try:
        version = importlib.metadata.version("hamiltonio")
    except importlib.metadata.PackageNotFoundError:
        version = None
    provenance = {
        "reader_module": getattr(module, "__file__", None),
        "reader_version": version,
        "wsvec_scheme2": "_gen_ham_scheme2" in source,
    }
    if not provenance["wsvec_scheme2"]:
        raise ValueError(
            "D5 HamiltonIO reader lacks the wsvec-aware _gen_ham_scheme2 Fourier"
            " transform required by the pinned use_ws_distance runs; the supported"
            " dependency is the TB2J_dev HamiltonIO checkout; refusing silent"
            f" fallback with reader{provenance}"
        )
    return provenance


def consume(folder, seed):
    frozen = json.loads((folder / "frozen-workflow.json").read_text())
    error = folder / f"{seed}.werr"
    if error.exists():
        raise ValueError(f"D4 {seed}: {error.read_text()}")
    output = folder / f"{seed}.wout"
    if not output.is_file():
        raise ValueError(f"D4 {output}: external localization output absent")
    text = output.read_text()
    for marker in (
        "disentanglement convergence criteria satisfied",
        "wannierisation convergence criteria satisfied",
    ):
        if marker not in text.lower():
            raise ValueError(f"D4 {output}: missing {marker}")
    for suffix in ("_centres.xyz", "_hr.dat"):
        if not (folder / f"{seed}{suffix}").is_file():
            raise ValueError(f"D5 missing real output {seed}{suffix}")
    from HamiltonIO.wannier import WannierHam
    from lawaf import W90Downfolder

    try:
        model = WannierHam.read_from_wannier_dir(str(folder), seed)
        provenance = hamiltonio_provenance(type(model))
        downfolder = W90Downfolder(folder=str(folder), prefix=seed)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise ValueError(
            f"D5 {seed}: HamiltonIO/W90Downfolder cannot read real outputs: {exc}"
        ) from exc
    kpoints = np.asarray(frozen["held_out_reduced_kpoints"])
    energies, _ = model.solve_all(kpoints)
    heldout_path = folder / f"gaas-{frozen['kind']}o_DS3_WFK.nc"
    if not heldout_path.is_file():
        raise ValueError(f"D2 {heldout_path}: held-out reference WFK absent")
    with WfkReader.open(heldout_path) as reference:
        if not 0 <= reference.header.residm <= frozen["nscf_residual_target"]:
            raise ValueError(f"D2 {heldout_path}: held-out wavefunctions not converged")
        np.testing.assert_allclose(reference.header.kpt, kpoints, atol=1e-14, rtol=0)
        expected = (
            np.stack(
                [reference.eigenvalues(k, 0, (10, 18)) for k in range(len(kpoints))]
            )
            * HA_EV
        )
    if np.asarray(energies).shape != expected.shape or not np.isfinite(energies).all():
        raise ValueError(
            f"D5 HamiltonIO dimensions/finite spectra differ: {np.asarray(energies).shape}, expected{expected.shape}"
        )
    difference = np.asarray(energies) - expected
    maximum = float(abs(difference).max())
    report = {
        "seed": seed,
        "held_out_original_bands": list(range(10, 18)),
        "held_out_error_eV": difference.tolist(),
        "held_out_max_error_eV": maximum,
        "HamiltonIO_functions": model.nbasis,
        "HamiltonIO_provenance": provenance,
        "W90Downfolder_reader": type(downfolder).__name__,
    }
    (folder / f"{seed}-consumer-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if maximum > 0.010:
        raise ValueError(
            f"held-out scalar interpolation{maximum:.6g}eV exceeds Architecture9 0.010eV gate"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("characterize", "win", "matrices", "consume")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pspdir", type=Path, required=True)
    parser.add_argument("--seed", default="gaas")
    args = parser.parse_args()
    folder, pspdir = args.run_dir.resolve(), args.pspdir.resolve()
    if args.action == "characterize":
        characterize(folder, pspdir)
    elif args.action == "consume":
        consume(folder, args.seed)
    else:
        generate(folder, pspdir, args.action == "matrices")
