from dataclasses import dataclass
from typing import Tuple, Union, List
import json
import yaml
import numpy as np

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
        return self.__dict__
    
    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)



