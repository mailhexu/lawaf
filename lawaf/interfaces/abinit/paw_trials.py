# SPDX-License-Identifier: BSD-2-Clause
"""Defining numerical PAW trials and local-closure admission (Architecture4.3).

`paw_augmented_partial_wave` explicitly chooses the pseudo partial wave inside
the dataset augmentation radius, zero outside; applying PAW yields the compact
AE partial-wave trial. Raw XML scattering waves extending far beyond that
radius are not localized trials. `paw_dual_projector` chooses the actual dual
projector and is admitted only when its defining transformed overlap agrees
with the PAW metric. Supplied electronic coefficient kinds remain distinct.

* ``__call__(ik, kg_rows)`` returns the plane-wave ket of the smooth trial
  Bloch sum, ``<exp(i(k+G)r) | trial>``: the Fourier transform of the
  numerical smooth trial with the atomic phase ``exp(-iq.tau_s)`` and the
  cell volume factor ``1/sqrt(V)`` applied exactly once.
* ``dual_coefficients(ik)`` returns the genuine one-center integrals
  ``<p_atom_i | Blochsum trial_t>``, i.e. ``sum_R exp(i k.R) * integral
  p_i(x)^* g_t(x - d_R) d^3x`` over ALL periodic images whose supports
  overlap, off-site images included.  Concentric geometries use the exact
  native-grid radial Gram times the harmonic-frame overlap; every other
  overlapping geometry uses converged direct 3D quadrature.  Generic
  non-overlap is never assumed silently: images are enumerated geometrically
  from the declared augmentation radii.

The geometry cache is static (species pair + displacement), bounded by the
enumerated neighbor geometry and independent of Nk.  No plane-wave or cprj
array is cached.  Scalar trial labels only; source-spin classification is
deliberately NOT rejected here (the producer owns that decision).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from lawaf.interfaces.abinit.basis_datasets import DatasetError
from lawaf.interfaces.abinit.nc_trials import real_harmonics
from lawaf.interfaces.abinit.paw_kernels import (
    HARMONIC_CONVENTION,
    _native_simpson_weights,
    _overint,
)

QUADRATURE_INTERNAL_GATE = 1e-10
PAW_GATE = 1e-8
_QUADRATURE_SCHEDULE = (20, 32, 48, 72, 104, 144)
_REFINED_STEP = 0.25
_CONCENTRIC_TOL = 1e-12


def _fail(file, field, expected, detail="") -> DatasetError:
    msg = f"{file}: field {field!r}: expected {expected}. {detail}".rstrip()
    return DatasetError(msg)




def _harmonic_frame_block(l, axes):
    """<Y_lm(xhat) | Y_lm'(xhat_local)> for one l; exact when frames coincide."""
    axes = np.asarray(axes, dtype=float)
    if np.allclose(axes, np.eye(3), atol=1e-12, rtol=0):
        return np.eye(2 * l + 1)
    costheta, weights = np.polynomial.legendre.leggauss(48)
    phi = np.arange(192) * 2 * np.pi / 192
    ct, ph = np.meshgrid(costheta, phi, indexing="ij")
    directions = np.column_stack(
        (
            np.sqrt(1 - ct.ravel() ** 2) * np.cos(ph.ravel()),
            np.sqrt(1 - ct.ravel() ** 2) * np.sin(ph.ravel()),
            ct.ravel(),
        )
    )
    angular_weights = np.repeat(weights, 192) * 2 * np.pi / 192
    left = real_harmonics(l, directions)
    right = real_harmonics(l, directions @ axes.T)
    return left.T @ (angular_weights[:, None] * right)


class PawTrialBasis:
    """Shared PAW trial interface: ``(ik, kg_rows)`` kets and dual overlap.

    ``header``/``kernels``/``offsets`` are the read-only matched provider
    data; trial columns carry scalar labels (site, actual XML radial state
    id, harmonic slot) in manifest order.
    """

    def __init__(self, manifest, provider):
        self.manifest = manifest
        self.provider = provider
        self.header = h = provider.header
        self.kernels = provider.kernels
        self.offsets = provider.offsets
        if manifest.trial_definition not in (
            "paw_dual_projector",
            "paw_augmented_partial_wave",
        ):
            raise _fail(
                manifest.path,
                "trial_definition",
                "explicit PAW radial trial definition",
                f"declared {manifest.trial_definition!r}",
            )
        self._trial_field = (
            "dual_projector"
            if manifest.trial_definition == "paw_dual_projector"
            else "pseudo_partial_wave"
        )
        self._compact_trial = manifest.trial_definition == "paw_augmented_partial_wave"
        self.closure_report = {}
        if manifest.units.get("length") != "bohr":
            raise _fail(manifest.path, "units.length", "bohr for PAW XML functions")
        convention = manifest.harmonic_convention
        if (
            convention.get("name") != HARMONIC_CONVENTION
            or convention.get("normalization") != "normalized"
        ):
            raise _fail(
                manifest.path,
                "harmonic_convention",
                f"normalized {HARMONIC_CONVENTION!r}",
            )
        if manifest.spin_frame_labels or any(
            record.spin_label is not None for record in manifest.trial_records
        ):
            raise _fail(
                manifest.path,
                "spin_label",
                "scalar spatial trial labels here; source-spin rejection "
                "belongs to the producer, not the trial basis",
            )
        if len(manifest.sites) != h.natom:
            raise _fail(
                manifest.path,
                "sites",
                f"one trial site per source atom ({h.natom})",
                f"declared {len(manifest.sites)}",
            )
        positions = np.asarray(h.xred, dtype=float) @ np.asarray(h.rprimd, float)
        declared = np.asarray([site.position for site in manifest.sites], float)
        if manifest.position_coordinates == "reduced":
            declared = declared @ np.asarray(h.rprimd, float)
        if not np.allclose(declared, positions, atol=1e-10, rtol=0):
            raise _fail(
                manifest.path,
                "sites.position",
                "trial sites at the actual source atom Cartesian positions (bohr)",
            )
        # Species mapping through the actual XML element of each matched kernel.
        self._site_species = np.empty(h.natom, dtype=int)
        for atom, site in enumerate(manifest.sites):
            kernel = self.kernels[int(h.typat[atom]) - 1]
            if site.species != kernel.dataset.element:
                raise _fail(
                    manifest.path,
                    f"sites[{atom}].species",
                    f"declared XML element {kernel.dataset.element!r} of the "
                    f"matched kernel for source atom {atom}",
                    f"declared {site.species!r}",
                )
            self._site_species[atom] = int(h.typat[atom]) - 1
        self._trial_columns = []
        for record in manifest.trial_records:
            species = self._site_species[record.site_index]
            kernel = self.kernels[species]
            channel_ids = [channel.state_id for channel in kernel.dataset.channels]
            if record.radial_id not in channel_ids:
                raise _fail(
                    manifest.path,
                    "radial_id",
                    f"an actual XML state id of species "
                    f"{record.species!r} ({channel_ids})",
                    f"declared {record.radial_id!r}",
                )
            channel_index = channel_ids.index(record.radial_id)
            channel = kernel.dataset.channels[channel_index]
            if (
                channel.l != record.l
                or not 0 <= record.harmonic_index < 2 * record.l + 1
            ):
                raise _fail(
                    manifest.path,
                    "radial/harmonic label",
                    f"actual XML l={channel.l} and a valid harmonic slot",
                )
            self._trial_columns.append(
                (record.site_index, species, channel_index, record.harmonic_index)
            )
        self.numspatialtrials = len(manifest.trial_records)
        self._admit_local_trial_closure()
        self._reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(h.rprimd, float)).T
        self._volume_factor = 1 / np.sqrt(abs(np.linalg.det(h.rprimd)))
        self._positions = positions
        self._axes = tuple(
            np.asarray(site.local_axes, float) for site in manifest.sites
        )
        self._support = tuple(kernel.dataset.rc for kernel in self.kernels)
        self._radial_splines = {}
        self._trial_splines = {}
        self._refined_measure = {}
        self._refined_r = {}
        self._refined_step = {}
        for species, kernel in enumerate(self.kernels):
            grid = kernel.dataset.grid
            self._radial_splines[species] = tuple(
                CubicSpline(grid.r, channel.dual_projector, extrapolate=False)
                for channel in kernel.dataset.channels
            )
            self._trial_splines[species] = tuple(
                CubicSpline(
                    grid.r, getattr(channel, self._trial_field), extrapolate=False
                )
                for channel in kernel.dataset.channels
            )
            # admitted refinement4 of the native log grid for the bounded
            # Fourier transform (same policy as the PawSpecies density work)
            indices = np.arange(len(grid.r))
            stop = (
                np.log1p(kernel.dataset.rc / grid.a) / grid.d
                if self._compact_trial
                else float(indices[-1])
            )
            intervals = int(np.ceil(stop / _REFINED_STEP))
            refined = np.linspace(0, stop, intervals + 1)
            self._refined_step[species] = stop / intervals
            refined_r = grid.a * np.expm1(grid.d * refined)
            refined_jacobian = grid.d * (refined_r + grid.a)
            self._refined_measure[species] = tuple(
                refined_r
                * refined_r
                * CubicSpline(indices, getattr(channel, self._trial_field))(refined)
                * refined_jacobian
                for channel in kernel.dataset.channels
            )
            self._refined_r[species] = refined_r
        self._neighbor_images = self._enumerate_images()
        # Static geometry cache: species pair + displacement; never Nk-keyed.
        self._geometry_cache = {}
        self.quadrature_report = {}

    def _admit_local_trial_closure(self):
        """Check defining Tpsi/Ttrial against the metric on local partial waves.

        Coefficient kind is always dualprojector; the trial radial function is
        a separate declaration. No Gram conversion of supplied electronic data.
        """
        for species, channel_index in sorted(
            {(species, channel) for _, species, channel, _ in self._trial_columns}
        ):
            kernel = self.kernels[species]
            grid = kernel.dataset.grid
            selected = kernel.dataset.channels[channel_index]
            channels = [
                channel
                for channel in kernel.dataset.channels
                if channel.l == selected.l
            ]
            ae = np.array([channel.ae_partial_wave for channel in channels]).T
            pseudo = np.array([channel.pseudo_partial_wave for channel in channels]).T
            projectors = np.array([channel.dual_projector for channel in channels]).T
            trial = getattr(selected, self._trial_field)[:, None]
            if self._compact_trial:
                trial = trial * (grid.r <= kernel.dataset.rc)[:, None]
            # Projector coefficients E and c use the m_paw_atom_solve
            # integrator on the native grid: per-integrand trailing
            # machine-zero trim (first following zero retained), Simpson
            # anchored at the trimmed last node, first trapezoid when even;
            # the radial Jacobian and r^2 stay explicit.
            measure = grid.r**2 * grid.d * (grid.r + grid.a)
            electronic = np.array(
                [
                    [
                        _overint(measure * projectors[:, k] * pseudo[:, m])
                        for m in range(pseudo.shape[1])
                    ]
                    for k in range(projectors.shape[1])
                ]
            )
            coefficients = np.array(
                [
                    [_overint(measure * projectors[:, k] * trial[:, 0])]
                    for k in range(projectors.shape[1])
                ]
            )
            # Defining, smooth and Delta-G overlaps use the common native
            # augmentation weights: m_pawrad simp_gen up to the nearest node
            # of the declared rc.
            augmentation = self._native_augmentation_weights(species)
            correction = ae - pseudo
            defining = (pseudo + correction @ electronic).T @ (
                augmentation[:, None] * (trial + correction @ coefficients)
            )
            smooth = pseudo.T @ (augmentation[:, None] * trial)
            delta_g = ae.T @ (augmentation[:, None] * ae) - pseudo.T @ (
                augmentation[:, None] * pseudo
            )
            metric = smooth + electronic.T @ delta_g @ coefficients
            residual = float(
                np.max(abs(defining - metric)) / max(1.0, float(np.max(abs(defining))))
            )
            self.closure_report[(species, selected.state_id)] = {
                "normalized_residual": residual,
                "gate": PAW_GATE,
            }
            if not np.isfinite(residual) or residual > PAW_GATE:
                raise _fail(
                    self.manifest.path,
                    "D5 trial local closure",
                    f"normalized defining PAW overlap residual <= {PAW_GATE}",
                    f"species{kernel.dataset.element} radial{selected.state_id} residual={residual}",
                )

    # ------------------------------------------------------------------
    # geometry: periodic images whose declared supports overlap
    # ------------------------------------------------------------------

    def _native_augmentation_weights(self, species):
        """Common m_pawrad simp_gen augmentation weights on the native grid.

        Support runs to the nearest node of the declared rc (the COMMON rpaw
        convention): Simpson anchored at that last node, first-node
        Cartwright correction (5/12, 2/3, -1/12) when the node count is
        even; the radial Jacobian and r^2 stay explicit in the weight.
        """
        dataset = self.kernels[species].dataset
        grid = dataset.grid
        nearest = int(np.argmin(np.abs(grid.r - dataset.rc))) + 1
        if nearest < 3:
            raise DatasetError(
                f"{dataset.path}: native augmentation requires at least three radial nodes"
            )
        weight = np.zeros(grid.r.shape[0])
        weight[:nearest] = (
            _native_simpson_weights(nearest)[::-1]
            * grid.d
            * (grid.r[:nearest] + grid.a)
            * grid.r[:nearest] ** 2
        )
        return weight

    def _enumerate_images(self):
        """{(atom, site): [(displacement d, Bloch image R), ...]} with overlap.

        Integer bounds are exact, not centered on zero: |d| <= cutoff with
        d = delta + n A implies n_i = (d.b_i - delta.b_i)/2pi, so
        |n_i + delta.b_i/2pi| <= cutoff |b_i| / 2pi.
        """
        h = self.header
        lattice = np.asarray(h.rprimd, float)
        reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        images = {}
        for atom in range(h.natom):
            for site in range(h.natom):
                delta = self._positions[site] - self._positions[atom]
                cutoff = (
                    self._support[self._site_species[atom]]
                    + self._support[self._site_species[site]]
                )
                center = -(delta @ reciprocal.T) / (2 * np.pi)
                radius = cutoff * np.linalg.norm(reciprocal, axis=1) / (2 * np.pi)
                bounds = [
                    (
                        int(np.ceil(center[axis] - radius[axis])),
                        int(np.floor(center[axis] + radius[axis])),
                    )
                    for axis in range(3)
                ]
                pairs = []
                for n1 in range(bounds[0][0], bounds[0][1] + 1):
                    for n2 in range(bounds[1][0], bounds[1][1] + 1):
                        for n3 in range(bounds[2][0], bounds[2][1] + 1):
                            n = np.array([n1, n2, n3], dtype=int)
                            displacement = delta + n @ lattice
                            if np.linalg.norm(displacement) <= cutoff:
                                pairs.append((displacement, n))
                images[atom, site] = tuple(pairs)
        return images

    # ------------------------------------------------------------------
    # defining overlap <p_species_a_i | trial species_s_j> at displacement d
    # ------------------------------------------------------------------

    def _radial_value(self, species, channel, points, *, trial=False):
        splines = self._trial_splines if trial else self._radial_splines
        radius = np.linalg.norm(points, axis=1)
        values = splines[species][channel](radius)
        if trial and self._compact_trial:
            values = np.where(radius <= self._support[species], values, 0.0)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _concentric_matrix(self, species_a, species_s, axes_s):
        """Radial Gram x harmonic frame for coincident centers.

        Production projector coefficients (<p_a,i | trial_s,j>) use the
        m_paw_atom_solve integrator: per-integrand trailing machine-zero
        trim with the first following zero retained, Simpson anchored at
        the trimmed last node, first trapezoid when even; the radial
        Jacobian and r^2 representation stay explicit.  The projector
        (atom) species carries the native grid, the trial radial function
        is read on it.
        """
        kernel_a, kernel_s = self.kernels[species_a], self.kernels[species_s]
        grid_a = kernel_a.dataset.grid
        jacobian_a = grid_a.d * (grid_a.r + grid_a.a)
        grid_s = kernel_s.dataset.grid
        # the projector (atom) species carries the native grid; the trial
        # radial function is read on it, cut at the trial rc when compact
        trial_values = {}
        for right_index, right in enumerate(kernel_s.dataset.channels):
            field = getattr(right, self._trial_field)
            if np.array_equal(grid_s.r, grid_a.r):
                trial_values[right_index] = field
            else:
                trial_values[right_index] = np.nan_to_num(
                    CubicSpline(grid_s.r, field, extrapolate=False)(grid_a.r),
                    nan=0.0,
                )
        radial = {}
        for left_index, left in enumerate(kernel_a.dataset.channels):
            for right_index, right in enumerate(kernel_s.dataset.channels):
                if left.l != right.l:
                    continue
                radial[(left.l, left_index, right_index)] = _overint(
                    left.dual_projector
                    * trial_values[right_index]
                    * grid_a.r
                    * grid_a.r
                    * jacobian_a
                    * (grid_a.r <= kernel_s.dataset.rc if self._compact_trial else 1.0)
                )
        frames = {}
        result = np.empty((len(kernel_a.basis), len(kernel_s.basis)))
        for row, (a_channel, a_slot) in enumerate(kernel_a.basis):
            l = kernel_a.dataset.channels[a_channel].l
            if l not in frames:
                frames[l] = _harmonic_frame_block(l, axes_s)
            for column, (s_channel, s_slot) in enumerate(kernel_s.basis):
                channel_s = kernel_s.dataset.channels[s_channel]
                value = 0.0
                if channel_s.l == l:
                    value = (
                        radial[(l, a_channel, s_channel)] * frames[l][a_slot, s_slot]
                    )
                result[row, column] = value
        return result

    def _quadrature_matrix(self, species_a, species_s, displacement, axes_s, nodes):
        """Direct 3D Gauss-Legendre quadrature over the projector support ball."""
        kernel_a, kernel_s = self.kernels[species_a], self.kernels[species_s]
        radius = self._support[species_a]
        edges, weights_1d = np.polynomial.legendre.leggauss(nodes)
        edges *= radius
        weights_1d *= radius
        axes_grid = np.meshgrid(edges, edges, edges, indexing="ij")
        points = np.stack(axes_grid, axis=-1).reshape(-1, 3)
        weights = (
            weights_1d[:, None, None]
            * weights_1d[None, :, None]
            * weights_1d[None, None, :]
        ).reshape(-1)
        nonvanishing = np.linalg.norm(points, axis=1) <= radius
        points, weights = points[nonvanishing], weights[nonvanishing]
        magnitude = np.linalg.norm(points, axis=1)
        direction = np.divide(
            points,
            magnitude[:, None],
            out=np.zeros_like(points),
            where=magnitude[:, None] != 0,
        )
        left_fields = np.empty((len(points), len(kernel_a.basis)))
        for column, (channel, slot) in enumerate(kernel_a.basis):
            l = kernel_a.dataset.channels[channel].l
            left_fields[:, column] = (
                self._radial_value(species_a, channel, points)
                * real_harmonics(l, direction)[:, slot]
            )
        relative = points - displacement
        magnitude = np.linalg.norm(relative, axis=1)
        direction = np.divide(
            relative,
            magnitude[:, None],
            out=np.zeros_like(relative),
            where=magnitude[:, None] != 0,
        )
        right_fields = np.empty((len(points), len(kernel_s.basis)))
        for column, (channel, slot) in enumerate(kernel_s.basis):
            l = kernel_s.dataset.channels[channel].l
            right_fields[:, column] = (
                self._radial_value(species_s, channel, relative, trial=True)
                * real_harmonics(l, direction @ axes_s.T)[:, slot]
            )
        return left_fields.T @ (weights[:, None] * right_fields)

    def _overlap_matrix(self, species_a, species_s, displacement, axes_s):
        """Cached defining overlap matrix; geometry-keyed, Nk-independent."""
        axes_key = tuple(np.asarray(axes_s, float).ravel())
        if np.linalg.norm(displacement) <= _CONCENTRIC_TOL:
            key = ("concentric", species_a, species_s, axes_key)
            if key not in self._geometry_cache:
                self._geometry_cache[key] = self._concentric_matrix(
                    species_a, species_s, axes_s
                )
            return self._geometry_cache[key]
        rounded = tuple(np.round(displacement, 12))
        key = ("quadrature", species_a, species_s, rounded, axes_key)
        if key not in self._geometry_cache:
            floor_key = ("concentric", species_a, species_s, axes_key)
            floor = float(
                np.max(
                    abs(
                        self._geometry_cache.setdefault(
                            floor_key,
                            self._concentric_matrix(species_a, species_s, axes_s),
                        )
                    )
                )
            )
            previous = None
            residual = None
            scale = None
            accepted = None
            for nodes in _QUADRATURE_SCHEDULE:
                current = self._quadrature_matrix(
                    species_a, species_s, displacement, axes_s, nodes
                )
                scale = max(float(np.max(abs(current))), floor)
                if previous is not None:
                    residual = float(np.max(abs(current - previous)))
                    if residual <= QUADRATURE_INTERNAL_GATE * scale:
                        accepted = QUADRATURE_INTERNAL_GATE
                        break
                previous = current
            else:
                # residual is the last successive level difference (the final
                # loop iteration must not zero it against itself)
                if residual is None or residual > PAW_GATE * scale:
                    raise _fail(
                        self.manifest.path,
                        "dual_coefficients",
                        f"direct 3D quadrature converged below the {PAW_GATE:.0e} "
                        f"PAW gate (internal target {QUADRATURE_INTERNAL_GATE:.0e})",
                        f"residual {residual} for displacement "
                        f"{np.asarray(displacement).tolist()}",
                    )
                accepted = PAW_GATE
            self.quadrature_report[key] = {
                "nodes": nodes,
                "residual": residual,
                "scale": scale,
                "gate": accepted,
            }
            self._geometry_cache[key] = current
        return self._geometry_cache[key]

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def dual_coefficients(self, ik):
        """<p_atom_i | Blochsum trial_t>(k): (total_projector, numspatialtrials)."""
        h = self.header
        if not 0 <= int(ik) < h.nkpt:
            raise _fail(self.manifest.path, "ik", f"a source k index in [0, {h.nkpt})")
        k_fractional = np.asarray(h.kpt[int(ik)], float)
        result = np.zeros((self.offsets[-1], self.numspatialtrials), complex)
        for column, (site, species_s, channel, slot) in enumerate(self._trial_columns):
            kernel_s = self.kernels[species_s]
            species_column = kernel_s.basis.index((channel, slot))
            for atom in range(h.natom):
                species_a = int(self._site_species[atom])
                for displacement, n in self._neighbor_images[atom, site]:
                    matrix = self._overlap_matrix(
                        species_a, species_s, displacement, self._axes[site]
                    )
                    # e^{i k.R} = e^{2 pi i f.n}, exact for any lattice
                    phase = np.exp(
                        2j * np.pi * float(k_fractional @ np.asarray(n, float))
                    )
                    rows = slice(self.offsets[atom], self.offsets[atom + 1])
                    result[rows, column] += phase * matrix[:, species_column]
        return result

    def __call__(self, ik, kg_rows):
        """Plane-wave ket <exp(i(k+G)r)|trial>/sqrt(V): (N, numspatialtrials)."""
        h = self.header
        if not 0 <= int(ik) < h.nkpt:
            raise _fail(self.manifest.path, "ik", f"a source k index in [0, {h.nkpt})")
        momentum = (np.asarray(kg_rows, float) + h.kpt[int(ik)]) @ self._reciprocal
        result = np.empty((len(momentum), self.numspatialtrials), complex)
        for start in range(0, len(momentum), 64):
            q = momentum[start : start + 64]
            magnitude = np.linalg.norm(q, axis=1)
            direction = np.divide(
                q,
                magnitude[:, None],
                out=np.zeros_like(q),
                where=magnitude[:, None] != 0,
            )
            radial = {}
            harmonics = {}
            phases = {}
            for column, (site, species, channel, slot) in enumerate(
                self._trial_columns
            ):
                kernel = self.kernels[species]
                l = kernel.dataset.channels[channel].l
                key = (species, channel)
                if key not in radial:
                    radial[key] = (
                        4
                        * np.pi
                        * (-1j) ** l
                        * simpson(
                            spherical_jn(
                                l, magnitude[:, None] * self._refined_r[species]
                            )
                            * self._refined_measure[species][channel],
                            dx=self._refined_step[species],
                            axis=1,
                        )
                    )
                angular_key = site, l
                if angular_key not in harmonics:
                    harmonics[angular_key] = real_harmonics(
                        l, direction @ self._axes[site].T
                    )
                if site not in phases:
                    phases[site] = np.exp(-1j * (q @ self._positions[site]))
                result[start : start + len(q), column] = (
                    self._volume_factor
                    * radial[key]
                    * harmonics[angular_key][:, slot]
                    * phases[site]
                )
        return result
