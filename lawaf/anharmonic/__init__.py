"""Anharmonic effective-model program for Lattice Wannier Functions.

Modules (see specs/architecture-lwf-anharmonic-effective-model.md):
- representation: S_g(q) builder, covariance + compatibility checks
- gauge: star-covariant constrained localization
- sampling: Q-space sampling sets, force projection
- dataset: TrainingDataset I/O
- teacher: calculator labeling, atomchain adapter, DFT spot checks
- basis: invariant polynomial basis (incl. strain sector)
- fit: residual-baseline joint fit
- model: evaluator + ASE calculator
- io: netCDF `symmetry` / `anharmonic` groups
"""
from lawaf.anharmonic.representation import (  # noqa: F401
    SeedsLabels,
    SpaceGroupAction,
    WindowBands,
    WindowLegalityBlock,
    build_space_group_action,
    check_window_legality,
)
from lawaf.anharmonic.compatibility import (  # noqa: F401
    CharacterLabels,
    CharacterTable,
    CompatCheck,
    RepresentationDeclaration,
    RepresentationLabels,
    RepresentationReport,
    assert_compatible,
    assert_covariance,
    character_table,
    check_compatibility,
    check_subspace_covariance,
    induced_characters,
    little_group,
    resolve_site_irrep,
)
from lawaf.anharmonic.sampling import (  # noqa: F401
    FrameSpec,
    SamplingPlan,
    make_atoms,
    project_forces,
    sample_frames,
)
from lawaf.anharmonic.gauge import (  # noqa: F401
    constrain_amn_one_q,
    constrain_builder_amn,
    constrained_localize,
    site_irrep_matrices,
)
from lawaf.anharmonic.dataset import (  # noqa: F401
    TrainingDataset,
)
from lawaf.anharmonic.teacher import (  # noqa: F401
    SpotCheckCategory,
    SpotCheckReport,
    from_abinit_hist,
    get_atomchain_calculator,
    label_frames,
    spot_check,
)
from lawaf.anharmonic.basis import (  # noqa: F401
    ClusterAction,
    ClusterCutoffs,
    ClusterKey,
    InvariantBasis,
    InvariantTerm,
    MolienReport,
    MolienRow,
    PermutationClusterAction,
    build_invariant_basis,
    build_oh_action,
    cluster_action_from_space_group,
    molien_check,
)
