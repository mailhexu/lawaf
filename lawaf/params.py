import copy
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import toml
import yaml


@dataclass
class WannierParams:
    """
    parameters for the wannierization.
    """

    method = "scdmk"
    kmesh: Tuple[int] = (2, 2, 2)
    kshift = np.array([0, 0, 0], dtype=float)
    kpts = None
    kweights = None
    gamma: bool = True
    nwann: int = 3
    weight_func: Union[None, str, callable] = "unity"
    weight_func_params: Union[None, dict] = None
    selected_basis: Union[None, List[int]] = None
    anchors: Union[None, List[int]] = None
    anchor_kpt: Tuple[int] = (0, 0, 0)
    anchor_ibands: Union[None, List[int]] = (0, 1, 2)
    use_proj: bool = True
    proj_order: int = 1
    exclude_bands: Tuple[int] = ()
    sort_cols: bool = True
    enhance_Amn: int = 0
    selected_orbdict = None
    orthogonal = True
    use_ws_distance: bool = True
    mlwf_tol: float = 1e-10
    mlwf_max_iter: int = 100
    mlwf_initial_guess: str = "projected"
    mlwf_fixed_gauge: bool = False
    # -- story-030: disentanglement selection windows (Epic 9, ADR-003) --
    dis_win_min: Optional[float] = None
    dis_win_max: Optional[float] = None
    dis_froz_min: Optional[float] = None
    dis_froz_max: Optional[float] = None
    dis_mix_ratio: float = 0.5
    dis_max_iter: int = 100
    dis_tol: float = 1e-10
    dis_min_svd: float = 1e-8
    dis_slow_tail_change: float = 1e-4
    symmetry_seed: bool = False
    symmetry_seed_opd: Union[None, int, dict] = None
    symmetry_seed_opd_index: Union[None, int, dict] = None
    window_bands: Optional[dict] = None
    # -- story-018: star-covariant constrained gauge (FR-004/015, ADR-003) --
    symmetry_adapted_gauge: bool = False
    representation_declaration: Union[None, dict] = None
    gauge_tolerances: Union[None, dict] = None

    # keys accepted for backward compatibility: forwarded through
    # Lawaf.downfold(**params) -> WannierParams.update by existing scripts
    # (mu/sigma feed weight_func_params; write_hr_* / post_func are consumed
    # by the calling script after downfold)
    _LEGACY_KEYS = frozenset(
        {"mu", "sigma", "write_hr_nc", "write_hr_txt", "post_func"}
    )

    @classmethod
    def _known_keys(cls):
        known = set()
        for key, value in cls.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if isinstance(value, (classmethod, staticmethod, property)):
                continue
            known.add(key)
        known |= set(cls.__annotations__)
        known |= set(cls._LEGACY_KEYS)
        return known

    def _check_unknown(self, kwargs):
        unknown = set(kwargs) - self._known_keys()
        if unknown:
            import warnings

            warnings.warn(
                f"Unknown WannierParams key(s) {sorted(unknown)}; "
                "possible typo. Known keys: "
                f"{sorted(self._known_keys())}",
                UserWarning,
                stacklevel=3,
            )

    def __init__(self, **kwargs):
        self._check_unknown(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, pdict):
        self._check_unknown(pdict)
        for key, value in pdict.items():
            setattr(self, key, value)

    def set(self, key, value):
        print(f"set {key} to {value}")
        setattr(self, key, value)
        if key == "anchors":
            self.anchor_kpt = list(value.keys())[0]
            self.anchor_ibands = value[self.anchor_kpt]
        if key == "anchor_kpt":
            self.anchors = {tuple(value): self.anchor_ibands}
        if key == "anchor_ibands":
            self.anchors = {tuple(self.anchor_kpt): value}
        print(self)

    def to_dict(self):
        mdict = copy.deepcopy(self.__dict__)
        for key, value in mdict.items():
            if isinstance(value, np.ndarray):
                mdict[key] = value.tolist()
        return mdict

    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    def to_toml(self, filename):
        with open(filename, "w") as f:
            toml.dump(self.to_dict(), f)

    @classmethod
    def from_toml(cls, filename):
        with open(filename, "r") as f:
            data = toml.load(f)
        return cls(**data)



def test_params():
    params = WannierParams(
        method="scdmk",
        kmesh=(5, 5, 5),
        kshift=np.array([1e-7, 3e-6, 5e-9]),
        gamma=True,
        nwann=0,
        weight_func="unity",
        weight_func_params=None,
        selected_basis=[9, 10, 11],
        anchor_kpt=(0, 0, 0),
        anchro_ibands=[0, 1, 2],
        use_proj=True,
        proj_order=1,
        exclude_bands=(),
        sort_cols=True,
        enhance_Amn=False,
        orthogonal=True,
    )
    print(params.to_dict())
    params.to_yaml("params.yaml")
    params.to_json("params.json")
    params.to_toml("params.toml")

    params=WannierParams.from_toml("params.toml")
    print(params.to_dict())


if __name__ == "__main__":
    test_params()
