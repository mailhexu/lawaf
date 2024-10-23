import copy
import json
from dataclasses import dataclass
from typing import List, Tuple, Union

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

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, pdict):
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
