import numpy as np
import torch
from typing import List

def num2str(num):
    return str(num).replace('.', 'd')[:5]

def vars2str(variables: List[str]):
    var_comb = ""
    for v in variables:
        var_comb += v + "_"
    var_comb = var_comb[:-1]
    return var_comb
