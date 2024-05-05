from dataclasses import dataclass
from typing import Tuple, Union, List
import json
import yaml
import numpy as np
import copy
import toml

@dataclass
class WannierParams:
    """
    parameters for the wannierization.
    """
    method = "scdmk"
    kmesh: Tuple[int] = (5, 5, 5)
    kshift = np.array([1e-7, 3e-6, 5e-9])
    kpts = None
    kweights = None
    gamma: bool = True
    nwann: int = 0
    weight_func: Union[None, str, callable] = "unity"
    weight_func_params: Union[None, dict] = None
    selected_basis: Union[None, List[int]] = None
    anchors: Union[None, List[int]] = None
    anchor_kpt: Tuple[int] = (0, 0, 0)
    anchor_ibands: Union[None, List[int]] = None
    use_proj: bool = True
    exclude_bands: Tuple[int] = ()
    sort_cols: bool = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, pdict):
        for key, value in pdict.items():
            setattr(self, key, value)
    
    def to_dict(self):
        mdict= copy.deepcopy(self.__dict__)
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
        with(open(filename, "w")) as f:
            toml.dump(self.to_dict(), f)


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
        exclude_bands=(),
        sort_cols=True
    )
    print(params.to_dict())
    params.to_yaml("params.yaml")
    params.to_json("params.json")
    params.to_toml("params.toml")

if __name__ == "__main__":
    test_params()
