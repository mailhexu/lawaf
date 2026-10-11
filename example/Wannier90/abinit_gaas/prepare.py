"""Prepare pinned scalar GaAs inputs; ABINIT and Wannier90 remain external."""

import argparse
import hashlib
import json
from pathlib import Path

PINS = {
    "Psdj_nc_fr_04_pbe_std_psp8/Ga.psp8": "38fe1b31919aae78137f937f249b5cb8c9d8e3cb131ea3ed5c61a42e5874867c",
    "Psdj_nc_fr_04_pbe_std_psp8/As.psp8": "78dfcd3f9e92c420d98368bc036cb702d3058e8252a2c63139fa3766ad6b9a52",
    "Psdj_nc_fr_04_pbe_std_psp8/Ga.upf": "e1aad2af1f8e69635634fd601d945943e033a36a1bb404cd4a1134e67180de7d",
    "Psdj_nc_fr_04_pbe_std_psp8/As.upf": "74e21823d94edd4762de345811a3ce82f282f2768debef657399a61cfae3c7a2",
    "Psdj_paw_pbe_std/Ga.xml": "10b9b72c90d5f758653efd89f5871bb43cf4e1e354e7683b8cbf65619e771ba4",
    "Psdj_paw_pbe_std/As.xml": "e04d9bb679cd9e63cf8ba1303bc285e785727cc83664d57a847422556ad06b40",
}
# Fixed before source/localization comparisons: valence and near-edge probes,
# including Gamma-X/Gamma-L directions and generic non-grid points.
HELD_OUT = (
    (0.02, 0, 0),
    (0.05, 0, 0),
    (0.10, 0.10, 0),
    (0.125, 0.125, 0.125),
    (0.075, 0.025, 0.05),
    (0.20, 0.05, 0),
    (0.08, 0.12, 0.16),
    (0.17, 0.07, 0.11),
)


def checked_datasets(pspdir, kind):
    prefix = "Psdj_paw_pbe_std/" if kind == "paw" else "Psdj_nc_fr_04_pbe_std_psp8/"
    checked = {}
    for name, expected in PINS.items():
        if not name.startswith(prefix):
            continue
        path = pspdir / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(
                f"D1 dataset {path}: expected SHA256{expected}, found{actual}"
            )
        checked[name] = actual
    return checked


def prepare(kind, pspdir, folder, mesh=4):
    pspdir, folder = Path(pspdir).resolve(), Path(folder).resolve()
    if folder.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Run artifacts must be outside the lawaf source repository")
    if (folder / "source-plan.json").exists() or (folder / f"gaas-{kind}.abi").exists():
        raise ValueError(
            f"D3 {folder}: existing source preparation must not be overwritten"
        )
    hashes = checked_datasets(pspdir, kind)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "Psdj_paw_pbe_std" if kind == "paw" else "Psdj_nc_fr_04_pbe_std_psp8"
    suffix = "xml" if kind == "paw" else "psp8"
    pseudopotentials = ", ".join(
        str(pspdir / prefix / f"{name}.{suffix}") for name in ("Ga", "As")
    )
    physics = (
        "ecut 20\npawecutdg 40\npawspnorb 0" if kind == "paw" else "ecut 40\nso_psp 1 1"
    )
    heldout = "\n".join(
        " ".join(f"{value:.17g}" for value in point) for point in HELD_OUT
    )
    deck = f"""# Architecture9 scalar first-example settings; residual stopping criteria.
ndtset 3
jdtset 1 2 3
acell 3*10.68
rprim 0 .5 .5  .5 0 .5  .5 .5 0
natom 2
ntypat 2
typat 1 2
znucl 31 33
xred 0 0 0  .25 .25 .25
pseudos "{pseudopotentials}"
{physics}
nband 24
nbdbuf 0
nspinor 1
nsppol 1
nspden 1
iomode 3
ngkpt {mesh} {mesh} {mesh}
nshiftk 1
shiftk 0 0 0
kptopt1 1
iscf1 7
tolvrs1 1e-12
nstep1 600
prtwf1 1
kptopt2 3
iscf2 -2
getden2 1
tolwfr2 1e-20
wfoptalg2 10
nline2 30
nstep2 400
istwfk2 *1
prtwf2 1
kptopt3 0
nkpt3 {len(HELD_OUT)}
kpt3
{heldout}
iscf3 -2
getden3 1
tolwfr3 1e-20
wfoptalg3 10
nline3 30
nstep3 400
istwfk3 *1
prtwf3 1
"""
    path = folder / f"gaas-{kind}.abi"
    path.write_text(deck)
    record = {
        "kind": kind,
        "mesh": [mesh] * 3,
        "dataset_sha256": hashes,
        "source_input": str(path),
        "scf_residual_target": 1e-12,
        "nscf_residual_target": 1e-20,
        "source_bands": 24,
        "retained_original_bands": list(range(10, 24)),
        "num_wann": 8,
        "held_out_reduced_kpoints": HELD_OUT,
        "held_out_target_original_bands": list(range(10, 18)),
        "manifold_certification": "requires characterization before window freeze/localization",
    }
    (folder / "source-plan.json").write_text(json.dumps(record, indent=2) + "\n")
    return path


def prepare_reuse(
    kind,
    pspdir,
    folder,
    source_dir,
    experiment_id,
    authorization,
    source_stdouts=None,
):
    """Authorize a new experiment that reuses an already converged source verbatim.

    No ABINIT run happens here: the independently converged full-mesh DS2 WFK and
    the fixed held-out DS3 WFK of ``source_dir`` are reused byte-identically via
    symlinks, and the convergence evidence is copied under the canonical names.
    """
    pspdir, folder, source_dir = (
        Path(p).resolve() for p in (pspdir, folder, source_dir)
    )
    if folder.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Run artifacts must be outside the lawaf source repository")
    if (folder / "source-plan.json").exists() or (folder / f"gaas-{kind}.abi").exists():
        raise ValueError(
            f"D3 {folder}: existing source preparation must not be overwritten"
        )
    base_path = source_dir / "source-plan.json"
    if not base_path.is_file():
        raise ValueError(f"D2 {base_path}: reused source plan absent")
    base = json.loads(base_path.read_text())
    if base["kind"] != kind:
        raise ValueError(
            f"D2 {base_path}: reused source kind{base['kind']} differs from requested{kind}"
        )
    checked_datasets(pspdir, kind)
    prefix = f"gaas-{kind}"
    deck = source_dir / f"{prefix}.abi"
    stdouts = [source_dir / name for name in (source_stdouts or (f"{prefix}.stdout",))]
    source_wfk = source_dir / f"{prefix}o_DS2_WFK.nc"
    heldout_wfk = source_dir / f"{prefix}o_DS3_WFK.nc"
    for path in (deck, source_wfk, heldout_wfk, *stdouts):
        if not path.is_file():
            raise ValueError(f"D2 {path}: reused source artifact absent")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{prefix}.abi").write_text(deck.read_text())
    for path in stdouts:
        (folder / path.name).write_text(path.read_text())
    for artifact in (source_wfk, heldout_wfk):
        (folder / artifact.name).symlink_to(artifact.resolve())
    record = {
        **base,
        "source_input": str(folder / f"{prefix}.abi"),
        "source_reuse_dir": str(source_dir),
        "source_reuse_deck": deck.name,
        "source_stdouts": [path.name for path in stdouts],
        "source_reuse": (
            "Independently converged full-mesh DS2 WFK and fixed held-out DS3 WFK"
            " reused byte-identically; no new ABINIT run for this experiment"
        ),
        "experiment_id": experiment_id,
        "user_authorization": authorization,
        "manifold_certification": "requires characterization before window freeze/localization",
    }
    plan = folder / "source-plan.json"
    plan.write_text(json.dumps(record, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("nc", "paw"), required=True)
    parser.add_argument("--pspdir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--reuse-source", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--authorization")
    parser.add_argument("--source-stdout", action="append")
    args = parser.parse_args()
    if args.reuse_source is not None:
        if not (args.experiment_id and args.authorization):
            parser.error("--reuse-source needs --experiment-id and --authorization")
        print(
            prepare_reuse(
                args.kind,
                args.pspdir,
                args.run_dir,
                args.reuse_source,
                args.experiment_id,
                args.authorization,
                source_stdouts=args.source_stdout,
            )
        )
    else:
        print(prepare(args.kind, args.pspdir, args.run_dir, args.mesh))
