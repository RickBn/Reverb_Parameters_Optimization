import os

import numpy as np
import pandas as pd

from scripts.utils.json_functions import *
from scripts.utils.plot_functions import *


def mape(reference, data, round_value: int = 1):

    if len(reference) != len(data):
        raise Exception("Both data arrays must have the same length.")

    diff = np.sum(np.abs([round(v, round_value) for v in (np.array(reference) - np.array(data)) / np.array(reference)]))
    mape = (diff / len(reference)) * 100

    return round(mape, round_value)



