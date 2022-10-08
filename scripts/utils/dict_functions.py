import os
import numpy as np
from typing import Dict

def get_dict_idx_value(d: Dict, idx: int):
	value = list(d.values())[idx]

	return value


def get_dict_idx_key(d: Dict, idx: int):
	key = list(d.keys())[idx]

	return key


def exclude_keys(d: Dict, keys):
	return {x: d[x] for x in d if x not in keys}


def save_or_merge(save_path: str, d: Dict, file_format: str = 'npy'):
	if file_format is 'npy':
		if os.path.exists(save_path):
			existing_dict = np.load(save_path, allow_pickle=True)[()]
			merged_dict = {**existing_dict, **d}
			np.save(save_path, merged_dict)
		else:
			np.save(save_path, d)