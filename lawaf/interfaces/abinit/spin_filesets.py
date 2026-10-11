# SPDX-License-Identifier: BSD-2-Clause
"""Separate physical-channel file sets; preprocessing/localization stay external."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

from lawaf.interfaces.abinit.producer import AbinitWannierInputs, ProducerError


class AbinitWannierFileSet:
    """One producer per actual channel; one physical-band set for a spinor.

    Plans and seeds are keyed by original zero-based source channel.
    Call write_win, run external -pp per seed, then pass each NNkp path to
    write_matrices. No executable or binding is owned by this object.
    """

    def __init__(
        self,
        source,
        plans,
        *,
        trial_orbitals=None,
        trial_manifest=None,
        paw_coefficients=None,
    ):
        channels = tuple(range(source.header.nsppol))
        if set(plans) != set(channels):
            raise ProducerError(
                "D1",
                source.path,
                "plan channels",
                f"source tuple{(source.header.nsppol,source.header.nspinor,source.header.nspden)} requires{channels}, found{tuple(plans)}",
            )
        producers = {}
        for channel in channels:
            plan = plans[channel]
            if plan.source_channel not in (None, channel):
                raise ProducerError(
                    "D1",
                    source.path,
                    "plan source_channel",
                    f"channel{channel}, found{plan.source_channel}",
                )
            record = None if paw_coefficients is None else paw_coefficients[channel]
            supplier = (
                trial_orbitals[channel]
                if isinstance(trial_orbitals, dict)
                else trial_orbitals
            )
            producers[channel] = AbinitWannierInputs(
                source,
                replace(plan, source_channel=channel),
                supplier,
                trial_manifest=trial_manifest,
                paw_coefficients=record,
            )
        self.producers = MappingProxyType(producers)

    def _seeds(self, seeds):
        if set(seeds) != set(self.producers):
            raise ProducerError(
                "D1", "<file-set>", "seed channels", f"exactly{tuple(self.producers)}"
            )
        if any(
            not isinstance(seed, str)
            or not seed
            or Path(seed).name != seed
            or seed in (".", "..")
            for seed in seeds.values()
        ):
            raise ProducerError(
                "D2",
                "<file-set>",
                "channel seed collision",
                "distinct filename stems, not path aliases",
            )
        if len(set(seeds.values())) != len(seeds):
            raise ProducerError(
                "D2",
                "<file-set>",
                "channel seed collision",
                f"distinct seeds required, found{seeds}",
            )
        return seeds

    def write_win(self, folder, seeds):
        seeds = self._seeds(seeds)
        return {
            channel: producer.write_win(folder, seeds[channel])
            for channel, producer in self.producers.items()
        }

    def write_matrices(self, nnkp_paths, folder, seeds):
        seeds = self._seeds(seeds)
        if set(nnkp_paths) != set(self.producers):
            raise ProducerError(
                "D1", "<file-set>", "NNkp channels", f"exactly{tuple(self.producers)}"
            )
        for channel, producer in self.producers.items():
            producer.validate_nnkp(nnkp_paths[channel])
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        manifests = {}
        with TemporaryDirectory(prefix=".channel-matrices-", dir=folder) as directory:
            for channel, producer in self.producers.items():
                manifests[channel] = producer.write_matrices(
                    nnkp_paths[channel], directory, seeds[channel]
                )
            for channel, manifest in manifests.items():
                files = {}
                for kind, path in manifest.files.items():
                    destination = folder / Path(path).name
                    Path(path).replace(destination)
                    files[kind] = str(destination)
                manifests[channel] = replace(manifest, files=files)
        return manifests
